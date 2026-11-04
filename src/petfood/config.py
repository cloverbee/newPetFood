from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://petfood:petfood@localhost:5432/petfood"

    s3_endpoint_url: str | None = "http://localhost:8333"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "petfood-papers"
    s3_region: str = "us-east-1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "petfood_chunks"

    # Retrieval models (fastembed / ONNX, run on CPU; downloaded on first use)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen3:8b"

    contact_email: str = "you@example.com"  # sent to Europe PMC / OPFF as polite User-Agent


@lru_cache
def get_settings() -> Settings:
    return Settings()
