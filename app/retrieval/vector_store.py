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
from sqlalchemy import and_, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import Document, DocumentChunk, DocumentType
from app.llm import make_embeddings

log = structlog.get_logger(__name__)


@dataclass
class RetrievalFilter:
    """Optional filters applied before hybrid relevance scoring."""

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
    similarity_score: Optional[float]
    retrieval_score: float = 0.0

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
    Combines pgvector cosine search and PostgreSQL full-text search.

    Candidate lists are merged with reciprocal-rank fusion, which avoids
    comparing incomparable vector-similarity and full-text rank scores.
    """

    def __init__(self) -> None:
        self._embeddings = None  # lazy-initialised on first use

    def _get_embeddings(self):
        """Return cached embeddings model, creating it on first call."""
        if self._embeddings is None:
            self._embeddings = make_embeddings()
        return self._embeddings

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        filters: Optional[RetrievalFilter] = None,
    ) -> list[RetrievedChunk]:
        """Return top-k chunks from fused semantic and lexical candidate lists."""
        if filters is None:
            filters = RetrievalFilter()

        top_k = filters.top_k or settings.similarity_top_k
        min_sim = (
            filters.min_similarity
            if filters.min_similarity is not None
            else settings.effective_similarity_threshold
        )
        candidate_limit = max(top_k * 3, 20)

        # Build the shared metadata filters.
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
        if filters.section_heading_contains:
            conditions.append(
                DocumentChunk.section_heading.ilike(f"%{filters.section_heading_contains}%")
            )

        # The same expression is indexed by the migration's GIN index.
        search_config = literal_column("'simple'::regconfig")
        searchable_text = (
            DocumentChunk.content
            + literal_column("' '")
            + func.coalesce(DocumentChunk.section_heading, literal_column("''"))
        )
        search_vector = func.to_tsvector(search_config, searchable_text)
        search_query = func.plainto_tsquery(search_config, query)

        # Semantic candidates.
        query_embedding = await self._get_embeddings().aembed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        similarity_expr = literal_column(
            f"1 - (embedding <=> CAST('{embedding_str}' AS vector))"
        )
        base_columns = (
            select(
                DocumentChunk,
                Document.company_ticker,
                Document.company_name,
                Document.document_type,
                Document.fiscal_year,
                Document.fiscal_quarter,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
        )

        vector_stmt = (
            base_columns
            .add_columns(similarity_expr.label("similarity"))
            .where(and_(similarity_expr >= min_sim, *conditions))
            .order_by(similarity_expr.desc())
            .limit(candidate_limit)
        )
        vector_result = await db.execute(vector_stmt)
        vector_chunks = [
            _to_retrieved_chunk(row, similarity_score=float(row.similarity))
            for row in vector_result.all()
        ]

        # Lexical candidates help with exact terms, tickers, and financial
        # vocabulary that may not be close in embedding space.
        lexical_score = func.ts_rank_cd(search_vector, search_query)
        lexical_stmt = (
            base_columns
            .add_columns(lexical_score.label("lexical_score"))
            .where(and_(search_vector.op("@@")(search_query), *conditions))
            .order_by(lexical_score.desc())
            .limit(candidate_limit)
        )
        lexical_result = await db.execute(lexical_stmt)
        lexical_chunks = [
            _to_retrieved_chunk(row, similarity_score=None)
            for row in lexical_result.all()
        ]

        chunks = reciprocal_rank_fusion(vector_chunks, lexical_chunks, top_k=top_k)

        log.info(
            "retrieval_complete",
            query_preview=query[:60],
            chunks_returned=len(chunks),
            semantic_candidates=len(vector_chunks),
            lexical_candidates=len(lexical_chunks),
        )
        return chunks


def _to_retrieved_chunk(row, *, similarity_score: Optional[float]) -> RetrievedChunk:
    return RetrievedChunk(
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
        similarity_score=similarity_score,
    )


def reciprocal_rank_fusion(
    semantic_chunks: list[RetrievedChunk],
    lexical_chunks: list[RetrievedChunk],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[RetrievedChunk]:
    """Merge ranked result lists using reciprocal-rank fusion."""
    fused: dict[int, RetrievedChunk] = {}
    scores: dict[int, float] = {}

    for ranked_chunks in (semantic_chunks, lexical_chunks):
        for rank, chunk in enumerate(ranked_chunks, start=1):
            existing = fused.get(chunk.chunk_id)
            if existing is None:
                fused[chunk.chunk_id] = chunk
            elif existing.similarity_score is None and chunk.similarity_score is not None:
                existing.similarity_score = chunk.similarity_score
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )

    ranked = sorted(fused.values(), key=lambda chunk: scores[chunk.chunk_id], reverse=True)
    for chunk in ranked:
        chunk.retrieval_score = scores[chunk.chunk_id]
    return ranked[:top_k]
