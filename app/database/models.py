"""
SQLAlchemy ORM models.

Design decisions:
  - `documents`        → metadata for each ingested report (company, period, type)
  - `document_chunks`  → text chunks with pgvector embeddings for semantic search
  - `financial_metrics`→ structured numerical data extracted from tables; queryable via SQL
  - `queries`          → audit log of every question + its final answer
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Enumerations ──────────────────────────────────────────────────────────────

class DocumentType(str, PyEnum):
    ANNUAL_REPORT = "annual_report"       # 10-K / Annual
    QUARTERLY_REPORT = "quarterly_report" # 10-Q / Quarterly
    EARNINGS_RELEASE = "earnings_release"
    RESEARCH_NOTE = "research_note"
    ECONOMIC_RELEASE = "economic_release"
    OTHER = "other"


class MetricCategory(str, PyEnum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    SEGMENT = "segment"
    GUIDANCE = "guidance"
    MACRO = "macro"


# ── Core tables ───────────────────────────────────────────────────────────────

class Document(Base):
    """One ingested financial report."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_ticker: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(200))
    document_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True,
    )
    fiscal_year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer)  # 1-4 or None
    period_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    filing_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    source_filename: Mapped[str] = mapped_column(String(500))
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    raw_text_length: Mapped[Optional[int]] = mapped_column(Integer)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["FinancialMetric"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} ticker={self.company_ticker} "
            f"type={self.document_type} fy={self.fiscal_year} q={self.fiscal_quarter}>"
        )


class DocumentChunk(Base):
    """
    A text segment from a document, with its pgvector embedding.

    `chunk_index` preserves reading order; `page_number` links back to the
    source page so citations can include page references.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    section_heading: Mapped[Optional[str]] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    # pgvector column — dimension matched to embedding model
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


class FinancialMetric(Base):
    """
    A single numerical financial data point extracted from a table or filing.

    Storing metrics in a relational table lets the agent answer
    time-series questions with SQL instead of relying on RAG.

    Examples:
        metric_name="revenue",        value=85_777.0, unit="USD_millions"
        metric_name="operating_margin", value=28.4,    unit="percent"
        metric_name="eps_diluted",    value=1.26,    unit="USD"
    """

    __tablename__ = "financial_metrics"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "metric_name", "segment",
            name="uq_metric_per_document_segment"
        ),
        Index("ix_metrics_ticker_name", "company_ticker", "metric_name"),
        Index("ix_metrics_period", "fiscal_year", "fiscal_quarter"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    company_ticker: Mapped[Optional[str]] = mapped_column(String(20), index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer)
    period_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    metric_name: Mapped[str] = mapped_column(String(200), nullable=False)
    metric_category: Mapped[MetricCategory] = mapped_column(
        Enum(MetricCategory, values_callable=lambda e: [m.value for m in e]),
        nullable=False
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(50))   # "USD_millions", "percent", etc.
    segment: Mapped[Optional[str]] = mapped_column(String(200))  # "Total" or business segment
    is_adjusted: Mapped[bool] = mapped_column(default=False)  # GAAP vs non-GAAP
    source_page: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="metrics")


class QueryLog(Base):
    """Audit record for every question answered by the agent."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    classification: Mapped[Optional[str]] = mapped_column(String(50))
    answer: Mapped[Optional[str]] = mapped_column(Text)
    citations: Mapped[Optional[dict]] = mapped_column(JSONB)
    tools_used: Mapped[Optional[list]] = mapped_column(JSONB)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
