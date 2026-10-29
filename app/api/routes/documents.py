"""Document catalogue API: inspect what's in the knowledge base."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import DocumentSummary, MetricSummary
from app.database.models import Document
from app.database.session import get_db
from app.retrieval.metric_store import MetricStore

router = APIRouter(prefix="/documents", tags=["documents"])
_metric_store = MetricStore()


@router.get("/", response_model=list[DocumentSummary])
async def list_documents(
    ticker: Optional[str] = None,
    document_type: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
) -> list[DocumentSummary]:
    """List all ingested documents, optionally filtered."""
    stmt = select(Document)
    if ticker:
        stmt = stmt.where(Document.company_ticker == ticker.upper())
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    if fiscal_year:
        stmt = stmt.where(Document.fiscal_year == fiscal_year)
    stmt = stmt.order_by(
        Document.company_ticker,
        Document.fiscal_year.desc(),
        Document.fiscal_quarter.desc().nullslast(),
    )

    result = await db.execute(stmt)
    docs = result.scalars().all()
    return [
        DocumentSummary(
            id=d.id,
            company_ticker=d.company_ticker,
            company_name=d.company_name,
            document_type=d.document_type,
            fiscal_year=d.fiscal_year,
            fiscal_quarter=d.fiscal_quarter,
            period_end_date=d.period_end_date,
            source_filename=d.source_filename,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> DocumentSummary:
    """Get a single document's metadata."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentSummary(
        id=doc.id,
        company_ticker=doc.company_ticker,
        company_name=doc.company_name,
        document_type=doc.document_type,
        fiscal_year=doc.fiscal_year,
        fiscal_quarter=doc.fiscal_quarter,
        period_end_date=doc.period_end_date,
        source_filename=doc.source_filename,
        created_at=doc.created_at,
    )


@router.get("/metrics/available", response_model=list[str])
async def list_available_metrics(
    ticker: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    """List all metric names extracted from filings."""
    return await _metric_store.list_available_metrics(db, ticker)


@router.get("/metrics/{ticker}", response_model=list[MetricSummary])
async def get_ticker_metrics(
    ticker: str,
    db: AsyncSession = Depends(get_db),
) -> list[MetricSummary]:
    """Get all metrics and their available periods for a specific ticker."""
    metrics = await _metric_store.list_available_metrics(db, ticker.upper())
    results = []
    for metric_name in metrics:
        periods = await _metric_store.list_available_periods(db, ticker.upper(), metric_name)
        results.append(MetricSummary(ticker=ticker.upper(), metric_name=metric_name, available_periods=periods))
    return results
