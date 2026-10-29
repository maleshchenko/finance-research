"""Initial schema: documents, chunks, metrics, query_logs.

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("company_ticker", sa.String(20), nullable=True, index=True),
        sa.Column("company_name", sa.String(200), nullable=True),
        sa.Column(
            "document_type",
            sa.Enum(
                "annual_report", "quarterly_report", "earnings_release",
                "research_note", "economic_release", "other",
                name="documenttype",
            ),
            nullable=False,
            index=True,
        ),
        sa.Column("fiscal_year", sa.Integer, nullable=True, index=True),
        sa.Column("fiscal_quarter", sa.Integer, nullable=True),
        sa.Column("period_end_date", sa.DateTime, nullable=True, index=True),
        sa.Column("filing_date", sa.DateTime, nullable=True),
        sa.Column("source_filename", sa.String(500), nullable=False),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column("raw_text_length", sa.Integer, nullable=True),
        sa.Column("extra_metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("document_id", sa.BigInteger, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("section_heading", sa.String(500), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )
    # HNSW index for fast approximate nearest-neighbour search
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "financial_metrics",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("document_id", sa.BigInteger, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("company_ticker", sa.String(20), nullable=True, index=True),
        sa.Column("fiscal_year", sa.Integer, nullable=False),
        sa.Column("fiscal_quarter", sa.Integer, nullable=True),
        sa.Column("period_end_date", sa.DateTime, nullable=True),
        sa.Column("metric_name", sa.String(200), nullable=False),
        sa.Column(
            "metric_category",
            sa.Enum(
                "income_statement", "balance_sheet", "cash_flow",
                "segment", "guidance", "macro",
                name="metriccategory",
            ),
            nullable=False,
        ),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("segment", sa.String(200), nullable=True),
        sa.Column("is_adjusted", sa.Boolean, default=False),
        sa.Column("source_page", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("document_id", "metric_name", "segment", name="uq_metric_per_document_segment"),
    )
    op.create_index("ix_metrics_ticker_name", "financial_metrics", ["company_ticker", "metric_name"])
    op.create_index("ix_metrics_period", "financial_metrics", ["fiscal_year", "fiscal_quarter"])

    op.create_table(
        "query_logs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("classification", sa.String(50), nullable=True),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("citations", sa.JSON, nullable=True),
        sa.Column("tools_used", sa.JSON, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("query_logs")
    op.drop_table("financial_metrics")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS documenttype")
    op.execute("DROP TYPE IF EXISTS metriccategory")
