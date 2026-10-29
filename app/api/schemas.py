"""Pydantic request/response schemas for the FastAPI surface."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    company_ticker: Optional[str] = Field(None, examples=["AAPL"])
    company_name: Optional[str] = Field(None, examples=["Apple Inc."])
    document_type: str = Field(
        "quarterly_report",
        examples=["quarterly_report", "annual_report", "earnings_release"],
    )
    fiscal_year: Optional[int] = Field(None, examples=[2024])
    fiscal_quarter: Optional[int] = Field(None, ge=1, le=4, examples=[3])
    period_end_date: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    extra_metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    metrics_created: int
    elapsed_seconds: float


# ── Research queries ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        examples=[
            "Compare operating margins for AAPL, MSFT, and GOOGL over the last four quarters.",
            "What forward guidance did Apple management provide in the Q3 2024 earnings call?",
        ],
    )


class ResearchResponse(BaseModel):
    answer: str
    classification: Optional[str]
    citations: list[str]
    tool_calls_made: list[str]
    elapsed_seconds: float
    error: Optional[str] = None


# ── Document catalogue ────────────────────────────────────────────────────────

class DocumentSummary(BaseModel):
    id: int
    company_ticker: Optional[str]
    company_name: Optional[str]
    document_type: str
    fiscal_year: Optional[int]
    fiscal_quarter: Optional[int]
    period_end_date: Optional[datetime]
    source_filename: str
    created_at: datetime


class MetricSummary(BaseModel):
    ticker: str
    metric_name: str
    available_periods: list[str]
