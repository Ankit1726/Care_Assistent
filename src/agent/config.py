from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    api_token: str = ""
    groq_api_key: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_index: str = "medical-agent"
    groq_chat_model: str = "openai/gpt-oss-20b"
    groq_vision_model: str = "qwen/qwen3.8-27b"
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("groq_chat_model", mode="before")
    @classmethod
    def replace_retired_chat_model(cls, model: str) -> str:
        if model == "llama-3.3-70b-versatile":
            return "openai/gpt-oss-20b"
        return model

    @field_validator("groq_vision_model", mode="before")
    @classmethod
    def replace_retired_vision_model(cls, model: str) -> str:
        if model == "llama-3.2-11b-vision-preview":
            return "qwen/qwen3.8-27b"
        return model


settings = Settings()