from functools import lru_cache
import math

from pinecone import Pinecone, ServerlessSpec

from src.agent.config import settings

EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PineconeIndexConfigurationError(ValueError):
    pass


@lru_cache
def embedding_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache
def clients():
    if not settings.pinecone_api_key:
        raise ValueError("Set PINECONE_API_KEY in .env")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(dimension: int = EMBEDDING_DIMENSION):
    pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        existing = pc.describe_index(settings.pinecone_index)
        if existing.dimension != dimension:
            raise PineconeIndexConfigurationError(
                f"Pinecone index '{settings.pinecone_index}' has dimension "
                f"{existing.dimension}; the configured embedding model requires "
                f"{dimension}. Use an index with dimension {dimension} and cosine metric."
            )
        if existing.metric != "cosine":
            raise PineconeIndexConfigurationError(
                f"Pinecone index '{settings.pinecone_index}' uses metric "
                f"'{existing.metric}'; configure it with cosine metric."
            )
    return pc.Index(settings.pinecone_index)


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    validated: list[list[float]] = []
    vectors = embedding_model().embed(texts)
    for vector in vectors:
        if getattr(vector, "ndim", 1) != 1:
            raise RuntimeError("Embedding service returned an invalid vector")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError):
            raise RuntimeError("Embedding service returned an invalid vector") from None
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("Embedding service returned an invalid vector")
        validated.append(values)

    if len(validated) != len(texts):
        raise RuntimeError("Embedding model returned an unexpected vector count")

    dimensions = {len(vector) for vector in validated}
    if len(dimensions) != 1:
        raise RuntimeError("Embedding service returned inconsistent vector dimensions")
    return validated


def search_kb(query: str, top_k: int = 4):
    vectors = embed([query])
    index = ensure_index(len(vectors[0]))
    result = index.query(vector=vectors[0], top_k=top_k, include_metadata=True)
    return [
        {
            "page": match.metadata.get("page"),
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 3),
            "source": "Gale Encyclopedia (2002)",
        }
        for match in result.matches
        if match.score >= 0.30
    ]
