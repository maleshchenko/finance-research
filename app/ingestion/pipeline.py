"""
Ingestion pipeline: PDF → chunks + metrics → database.

Usage:
    result = await ingest_document(
        db=session,
        file_path=Path("reports/AAPL_10Q_Q3_2024.pdf"),
        metadata=DocumentMetadata(
            company_ticker="AAPL",
            company_name="Apple Inc.",
            document_type=DocumentType.QUARTERLY_REPORT,
            fiscal_year=2024,
            fiscal_quarter=3,
        ),
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import (
    Document,
    DocumentChunk,
    DocumentType,
    FinancialMetric,
    MetricCategory,
)
from app.ingestion.chunker import chunk_document
from app.ingestion.extractor import extract_pdf
from app.llm import make_embeddings

log = structlog.get_logger(__name__)


@dataclass
class DocumentMetadata:
    company_ticker: Optional[str] = None
    company_name: Optional[str] = None
    document_type: DocumentType = DocumentType.OTHER
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    period_end_date: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    source_uri: Optional[str] = None          # s3:// URI or original filename
    extra_metadata: Optional[dict] = None


@dataclass
class IngestionResult:
    document_id: int
    chunks_created: int
    metrics_created: int
    elapsed_seconds: float


async def ingest_document(
    db: AsyncSession,
    file_path: Path,
    metadata: DocumentMetadata,
) -> IngestionResult:
    """
    Full ingestion pipeline:
      1. Extract text and tables from PDF.
      2. Chunk narrative text.
      3. Embed chunks in batches.
      4. Persist document, chunks, and metrics to the database.
    """
    t0 = time.perf_counter()
    log.info("ingestion_start", file=str(file_path), ticker=metadata.company_ticker)

    # ── 1. Extract ────────────────────────────────────────────────────────────
    extraction = extract_pdf(file_path)

    # ── 2. Create document record ─────────────────────────────────────────────
    doc = Document(
        company_ticker=metadata.company_ticker,
        company_name=metadata.company_name,
        document_type=metadata.document_type,
        fiscal_year=metadata.fiscal_year,
        fiscal_quarter=metadata.fiscal_quarter,
        period_end_date=metadata.period_end_date,
        filing_date=metadata.filing_date,
        source_filename=file_path.name,
        page_count=extraction.page_count,
        raw_text_length=extraction.raw_text_length,
        extra_metadata=metadata.extra_metadata,
    )
    db.add(doc)
    await db.flush()  # get doc.id without committing

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    chunks = chunk_document(extraction)
    texts = [c.content for c in chunks]

    # ── 4. Embed (batched to respect rate limits) ─────────────────────────────
    embeddings_model = make_embeddings()
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = await embeddings_model.aembed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        log.debug("embedded_batch", start=i, end=i + len(batch))

    # ── 5. Persist chunks ─────────────────────────────────────────────────────
    db_chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            section_heading=chunk.section_heading,
            content=chunk.content,
            token_count=chunk.token_count,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, all_embeddings, strict=True)
    ]
    db.add_all(db_chunks)

    # ── 6. Persist metrics ────────────────────────────────────────────────────
    seen: set[tuple] = set()
    db_metrics: list[FinancialMetric] = []
    for raw in extraction.metrics:
        key = (doc.id, raw.metric_name, raw.segment)
        if key in seen:
            continue
        seen.add(key)
        try:
            category = MetricCategory(raw.metric_category)
        except ValueError:
            category = MetricCategory.INCOME_STATEMENT

        db_metrics.append(
            FinancialMetric(
                document_id=doc.id,
                company_ticker=metadata.company_ticker,
                fiscal_year=metadata.fiscal_year or 0,
                fiscal_quarter=metadata.fiscal_quarter,
                period_end_date=metadata.period_end_date,
                metric_name=raw.metric_name,
                metric_category=category,
                value=raw.value,
                unit=raw.unit,
                segment=raw.segment,
                is_adjusted=raw.is_adjusted,
                source_page=raw.page_number,
            )
        )
    if db_metrics:
        db.add_all(db_metrics)

    await db.commit()

    elapsed = time.perf_counter() - t0
    log.info(
        "ingestion_complete",
        document_id=doc.id,
        chunks=len(db_chunks),
        metrics=len(db_metrics),
        elapsed=f"{elapsed:.1f}s",
    )

    return IngestionResult(
        document_id=doc.id,
        chunks_created=len(db_chunks),
        metrics_created=len(db_metrics),
        elapsed_seconds=elapsed,
    )
