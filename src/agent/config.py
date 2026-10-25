import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    groq_api_key: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_index: str = "medical-agent"
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_embed_model: str = "all-MiniLM-L6-v2"
    groq_vision_model: str = "llama-3.2-11b-vision-preview"
    hf_token: str = ""
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()