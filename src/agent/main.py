import base64
import logging
import secrets
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

# Allow running this file directly: python src/agent/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import BadRequestError, Groq
from pydantic import BaseModel, Field

from src.agent.backend import workflow
from src.agent.config import settings
from src.agent.ingest import SUPPORTED, ingest_file
from src.agent.knowledge import PineconeIndexConfigurationError

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 5_000_000
MAX_IMAGE_BASE64_LENGTH = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_AUDIO_BYTES = 10_000_000
MAX_UPLOAD_BYTES = 25_000_000

ROOT = Path(__file__).resolve().parent.parent  # -> src/
UI_DIR = ROOT / "ui"
STATIC_DIR = UI_DIR / "static"
# works whether your folder is named "template" or "templates"
TEMPLATE_DIR = next(
    (p for p in (UI_DIR / "template", UI_DIR / "templates") if p.exists()),
    UI_DIR / "template",
)

app = FastAPI(title="Care Health System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)
    image_data: str | None = Field(
        default=None, max_length=MAX_IMAGE_BASE64_LENGTH + 64
    )


def auth(authorization: str | None) -> None:
    expected = settings.api_token
    if not expected:
        return

    parts = (authorization or "").split(maxsplit=1)
    if (
        len(parts) != 2
        or parts[0].lower() != "bearer"
        or not secrets.compare_digest(
            parts[1].encode("utf-8"), expected.encode("utf-8")
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid API token")


def inspect_image(data: str | None) -> str:
    if not data:
        return ""
    try:
        header, separator, payload = data.partition(",")
        mime = header.removeprefix("data:")
        if (
            not separator
            or not header.startswith("data:")
            or not header.endswith(";base64")
            or mime.removesuffix(";base64")
            not in ("image/jpeg", "image/png", "image/webp")
            or len(payload) > MAX_IMAGE_BASE64_LENGTH
        ):
            raise ValueError("Only base64 JPEG/PNG/WebP images under 5MB are supported")
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Only base64 JPEG/PNG/WebP images under 5MB are supported")
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    client = Groq(api_key=settings.groq_api_key, timeout=45)
    try:
        result = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.",
                        },
                        {"type": "image_url", "image_url": {"url": data}},
                    ],
                }
            ],
            max_tokens=300,
        )
    except BadRequestError as exc:
        error = exc.body
        details = error.get("error") if isinstance(error, dict) else None
        message = details.get("message", "") if isinstance(details, dict) else ""
        if "at least 32 pixels in each dimension" in message:
            raise HTTPException(
                status_code=400,
                detail="Image is too small. Use an image at least 32 pixels wide and high.",
            ) from None
        raise
    return result.choices[0].message.content or ""


@app.get("/", response_class=HTMLResponse)
def home():
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    missing = [
        name
        for name, configured in (
            ("GROQ_API_KEY", settings.groq_api_key),
            ("PINECONE_API_KEY", settings.pinecone_api_key),
        )
        if not configured
    ]
    return {
        "status": "ok",
        "configured": not missing,
        "missing": missing,
        "pinecone_index": settings.pinecone_index,
        "pinecone_dimension": 384,
        "pinecone_metric": "cosine",
        "pinecone_region": "aws / us-east-1",
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    auth(authorization)
    if not settings.groq_api_key:
        raise HTTPException(503, "Set GROQ_API_KEY in .env")
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(400, "Audio missing or larger than 10MB")
    try:
        client = Groq(api_key=settings.groq_api_key, timeout=45)
        result = client.audio.transcriptions.create(
            file=(file.filename or "voice.webm", audio),
            model="whisper-large-v3-turbo",
        )
        return {"text": result.text}
    except Exception as exc:
        logger.error("Transcription failed (%s)", type(exc).__name__)
        raise HTTPException(502, "Voice transcription failed. Check server logs.")


@app.post("/api/ingest")
def ingest(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    auth(authorization)
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(
            503, "Set GROQ_API_KEY and PINECONE_API_KEY in .env"
        )
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(
            400, f"Unsupported file. Use: {', '.join(sorted(SUPPORTED))}"
        )
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File is empty or larger than 25MB")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        try:
            chunks = ingest_file(path)
        except PineconeIndexConfigurationError as exc:
            logger.error("Pinecone index configuration mismatch")
            raise HTTPException(503, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            logger.error("Document indexing failed (%s)", type(exc).__name__)
            raise HTTPException(
                502, "Indexing failed. Check server logs and Pinecone setup."
            )
    return {"file": name, "chunks": chunks}


@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias="X-Conversation-ID"),
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(
            503, "Set GROQ_API_KEY and PINECONE_API_KEY in .env"
        )

    history = [
        x
        for x in payload.history
        if x.get("role") in ("user", "assistant") and isinstance(x.get("content"), str)
    ][:10]

    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation = inspect_image(payload.image_data)

        state = workflow.invoke(
            {
                "question": payload.message,
                "history": history,
                "image_description": observation,
                "kb": [],
                "web": [],
                "answer": "",
                "route": "",
            },
            config=config,
        )

        return {
            "answer": state["answer"],
            "sources": [
                {"type": "pdf", "page": x["page"], "score": x["score"]}
                for x in state["kb"]
            ]
            + [
                {"type": "web", "url": x["url"], "title": x["title"]}
                for x in state["web"]
            ],
            "tools": ["pinecone"]
            + (
                ["tavily"]
                if state["route"] == "web" and settings.tavily_api_key
                else []
            )
            + (["vision"] if observation else []),
            "conversation_id": thread_id,
        }

    except HTTPException:
        raise
    except PineconeIndexConfigurationError as exc:
        logger.error("Pinecone index configuration mismatch")
        raise HTTPException(503, str(exc)) from None
    except Exception as exc:
        logger.error("Chat service failed (%s)", type(exc).__name__)
        raise HTTPException(
            502,
            "AI service failed. Verify API credentials, local embedding model, and the Pinecone index name, dimension, metric, and region.",
        )


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    url = "http://127.0.0.1:8000"
    print(f"\nCare Health System running at {url}\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)