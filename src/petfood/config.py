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
    rerank_max_words: int | None = None   # reranker reads only the first N words of each candidate (D21)
    # Where fastembed keeps downloaded models (~1.3 GB). Its default is a macOS temp folder that
    # the system may clear, forcing a re-download; a fixed folder avoids that.
    fastembed_cache_dir: str = "~/.cache/fastembed"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen3:4b"
    local_think: bool = False   # Qwen3 "thinking" mode: ~4x slower, rarely needed with a JSON schema
    # Context window in tokens (prompt + answer). Ollama's default of 4096 is nearly filled by an
    # answer prompt (~3.8k tokens); past it, Ollama silently drops the start, where the rules are.
    local_num_ctx: int = 8192

    contact_email: str = "you@example.com"  # sent to Europe PMC / OPFF as polite User-Agent


@lru_cache
def get_settings() -> Settings:
    return Settings()
