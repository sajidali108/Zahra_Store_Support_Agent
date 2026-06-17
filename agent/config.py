from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str = "your_groq_api_key_here"
    groq_model: str = "openai/gpt-oss-120b"
    shopify_api_key: str = "your_shopify_api_key_here"
    shopify_access_token: str = "your_shopify_access_token_here"
    shopify_store_url: str = "https://zahrastores.pk"
    shopify_api_version: str = "2024-10"
    zahra_api_base_url: str = "https://apis.zahrahomes.pk/api"
    zahra_api_token: str = "your_zahra_api_token_here"
    faiss_index_path: str = "rag/faiss_index"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
