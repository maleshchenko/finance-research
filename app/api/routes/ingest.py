"""Ingestion API: upload a PDF and trigger the ingestion pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import IngestRequest, IngestResponse
from app.config import settings
from app.database.models import DocumentType
from app.database.session import get_db
from app.ingestion.pipeline import DocumentMetadata, ingest_document

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    ticker: str | None = None,
    company_name: str | None = None,
    document_type: str = "quarterly_report",
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Upload a PDF financial document and ingest it into the knowledge base.

    The file is streamed to a temporary location, extracted, chunked,
    embedded, and stored. Structured metrics are extracted automatically.

    Accepts: PDF files up to `MAX_UPLOAD_SIZE_MB` MB.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # Validate size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    # Validate document type
    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        valid = [dt.value for dt in DocumentType]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid document_type. Valid values: {valid}",
        )

    # Write to temp file and run pipeline
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        metadata = DocumentMetadata(
            company_ticker=ticker,
            company_name=company_name,
            document_type=doc_type,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        result = await ingest_document(db, tmp_path, metadata)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        document_id=result.document_id,
        chunks_created=result.chunks_created,
        metrics_created=result.metrics_created,
        elapsed_seconds=round(result.elapsed_seconds, 2),
    )
