"""Application configuration loaded from environment variables."""

from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Provider selection ────────────────────────────────────────────────────
    # Set LLM_PROVIDER=ollama to use a local Ollama instance instead of OpenAI.
    llm_provider: Literal["openai", "ollama"] = "openai"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Ollama ────────────────────────────────────────────────────────────────
    # Run `ollama pull <model>` before use.
    # nomic-embed-text produces 768-dim vectors; set VECTOR_DIMENSION=768 to match.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/financial_research"
    database_pool_size: int = 5   # keep low — multiply by replica count
    database_max_overflow: int = 5

    # ── S3 (optional) ─────────────────────────────────────────────────────────
    s3_bucket: str = Field(default="")
    aws_region: str = Field(default="us-east-1")

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8080")

    # ── Vector store ──────────────────────────────────────────────────────────
    # Use 768 when ollama_embedding_model=nomic-embed-text; 1536 for OpenAI.
    vector_dimension: int = 1536
    similarity_top_k: int = 8
    similarity_threshold: float = 0.35

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_upload_size_mb: int = 50

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_max_iterations: int = 12
    agent_temperature: float = 0.1

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("openai_api_key")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _check_provider_config(self) -> "Settings":
        if self.llm_provider == "openai":
            if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                    "Copy .env.example to .env and fill it in, or set "
                    "LLM_PROVIDER=ollama to use a local model."
                )
        return self

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


settings = Settings()
