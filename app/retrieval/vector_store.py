"""
Semantic retrieval over document chunks using pgvector.

Supports metadata filtering on:
  - company_ticker   (exact or list)
  - document_type    (exact or list)
  - fiscal_year      (exact or range)
  - fiscal_quarter   (exact)
  - section_heading  (substring)

Returns `RetrievedChunk` objects that carry enough provenance for citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import Document, DocumentChunk, DocumentType

log = structlog.get_logger(__name__)


@dataclass
class RetrievalFilter:
    """Optional filters applied before cosine similarity scoring."""

    tickers: Optional[list[str]] = None
    document_types: Optional[list[DocumentType]] = None
    fiscal_year_min: Optional[int] = None
    fiscal_year_max: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    section_heading_contains: Optional[str] = None
    top_k: Optional[int] = None        # override global default
    min_similarity: Optional[float] = None  # override global default


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    company_ticker: Optional[str]
    company_name: Optional[str]
    document_type: str
    fiscal_year: Optional[int]
    fiscal_quarter: Optional[int]
    page_number: Optional[int]
    section_heading: Optional[str]
    content: str
    similarity_score: float

    # Convenience property used in prompt assembly
    @property
    def citation_label(self) -> str:
        parts = []
        if self.company_ticker:
            parts.append(self.company_ticker)
        if self.fiscal_year:
            q = f"Q{self.fiscal_quarter}" if self.fiscal_quarter else "Annual"
            parts.append(f"FY{self.fiscal_year} {q}")
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        return " · ".join(parts) if parts else f"doc:{self.document_id}"


class VectorRetriever:
    """
    Wraps pgvector cosine-distance search with metadata pre-filtering.

    The query is embedded, then we use the `<=>` operator (cosine distance)
    via a raw SQL expression because SQLAlchemy doesn't expose pgvector
    operators natively.
    """

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        filters: Optional[RetrievalFilter] = None,
    ) -> list[RetrievedChunk]:
        """Embed `query`, apply filters, return top-k chunks by cosine similarity."""
        if filters is None:
            filters = RetrievalFilter()

        top_k = filters.top_k or settings.similarity_top_k
        min_sim = filters.min_similarity or settings.similarity_threshold

        # Embed the query
        query_embedding = await self._embeddings.aembed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Build filter conditions on Document
        conditions = []
        if filters.tickers:
            conditions.append(Document.company_ticker.in_(filters.tickers))
        if filters.document_types:
            type_values = [dt.value for dt in filters.document_types]
            conditions.append(Document.document_type.in_(type_values))
        if filters.fiscal_year_min is not None:
            conditions.append(Document.fiscal_year >= filters.fiscal_year_min)
        if filters.fiscal_year_max is not None:
            conditions.append(Document.fiscal_year <= filters.fiscal_year_max)
        if filters.fiscal_quarter is not None:
            conditions.append(Document.fiscal_quarter == filters.fiscal_quarter)

        # Cosine similarity = 1 - cosine_distance
        similarity_expr = text(
            f"1 - (embedding <=> CAST(:embedding AS vector))"
        ).bindparams(embedding=embedding_str)

        stmt = (
            select(
                DocumentChunk,
                Document.company_ticker,
                Document.company_name,
                Document.document_type,
                Document.fiscal_year,
                Document.fiscal_quarter,
                similarity_expr.label("similarity"),
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                and_(
                    *conditions,
                    similarity_expr >= min_sim,
                )
            )
            .order_by(similarity_expr.desc())
            .limit(top_k)
        )

        result = await db.execute(stmt)
        rows = result.all()

        chunks = [
            RetrievedChunk(
                chunk_id=row.DocumentChunk.id,
                document_id=row.DocumentChunk.document_id,
                company_ticker=row.company_ticker,
                company_name=row.company_name,
                document_type=row.document_type,
                fiscal_year=row.fiscal_year,
                fiscal_quarter=row.fiscal_quarter,
                page_number=row.DocumentChunk.page_number,
                section_heading=row.DocumentChunk.section_heading,
                content=row.DocumentChunk.content,
                similarity_score=float(row.similarity),
            )
            for row in rows
        ]

        log.info(
            "retrieval_complete",
            query_preview=query[:60],
            chunks_returned=len(chunks),
        )
        return chunks
