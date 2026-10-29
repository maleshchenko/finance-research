"""Application configuration loaded from environment variables."""

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/financial_research"
    database_pool_size: int = 5   # keep low — multiply by replica count
    database_max_overflow: int = 5

    # S3 (optional — if set, PDFs are stored in S3 instead of local temp files)
    s3_bucket: str = Field(default="")
    aws_region: str = Field(default="us-east-1")

    # CORS — comma-separated list; use "*" only in local dev
    cors_allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8080")

    @field_validator("openai_api_key")
    @classmethod
    def openai_key_must_be_set(cls, v: str) -> str:
        if not v or not v.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY is required and must start with 'sk-'. "
                "Copy .env.example to .env and fill it in."
            )
        return v

    # Vector store
    vector_dimension: int = 1536
    similarity_top_k: int = 8
    similarity_threshold: float = 0.35

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_upload_size_mb: int = 50

    # Agent
    agent_max_iterations: int = 12
    agent_temperature: float = 0.1

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """Synchronous URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


settings = Settings()
