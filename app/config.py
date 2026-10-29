"""Application configuration loaded from environment variables."""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/financial_research"
    database_pool_size: int = 10
    database_max_overflow: int = 20

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
