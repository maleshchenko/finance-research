"""
Structured metric queries.

This module answers numerical time-series questions deterministically via SQL
instead of asking the LLM to reason over retrieved text.

Key insight from the architecture:
  RAG is for narrative / qualitative retrieval.
  SQL is for "what was the operating margin every quarter for 8 quarters?".

Using LangChain's `QuerySQLDatabaseChain` for natural-language-to-SQL is
intentionally avoided here — we prefer hand-crafted parameterised queries
that guarantee correctness and prevent SQL injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document, FinancialMetric

log = structlog.get_logger(__name__)


@dataclass
class MetricRecord:
    company_ticker: str
    fiscal_year: int
    fiscal_quarter: Optional[int]
    period_end_date: Optional[datetime]
    metric_name: str
    value: float
    unit: Optional[str]
    segment: str
    is_adjusted: bool
    document_id: int
    source_page: Optional[int]
    source_provider: str | None = None
    source_accessions: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    source_note: str | None = None

    @property
    def period_label(self) -> str:
        if self.fiscal_quarter:
            return f"FY{self.fiscal_year} Q{self.fiscal_quarter}"
        return f"FY{self.fiscal_year}"


def _prefer_sec_records(
    records: list[MetricRecord],
    sec_periods: set[tuple[str, int, int | None]],
) -> list[MetricRecord]:
    return [
        record
        for record in records
        if (record.company_ticker, record.fiscal_year, record.fiscal_quarter)
        not in sec_periods
        or record.source_provider == "sec_companyfacts"
    ]


class MetricStore:
    """
    Typed query interface for the `financial_metrics` table.
    """

    async def query_metric_series(
        self,
        db: AsyncSession,
        tickers: list[str],
        metric_names: list[str],
        fiscal_year_min: Optional[int] = None,
        fiscal_year_max: Optional[int] = None,
        fiscal_quarter: Optional[int] = None,
        segment: str = "Total",
        include_adjusted: bool = False,
    ) -> list[MetricRecord]:
        """
        Retrieve a time series of one or more metrics for one or more companies.

        Example:
            await store.query_metric_series(
                db,
                tickers=["AAPL", "MSFT", "GOOGL"],
                metric_names=["operating_margin", "revenue"],
                fiscal_year_min=2022,
                fiscal_year_max=2024,
            )
        """
        conditions = [
            FinancialMetric.company_ticker.in_(tickers),
            FinancialMetric.metric_name.in_(metric_names),
        ]
        if fiscal_year_min is not None:
            conditions.append(FinancialMetric.fiscal_year >= fiscal_year_min)
        if fiscal_year_max is not None:
            conditions.append(FinancialMetric.fiscal_year <= fiscal_year_max)
        if fiscal_quarter is not None:
            conditions.append(FinancialMetric.fiscal_quarter == fiscal_quarter)
        if segment != "all":
            conditions.append(FinancialMetric.segment == segment)
        if not include_adjusted:
            conditions.append(FinancialMetric.is_adjusted == False)  # noqa: E712

        stmt = (
            select(FinancialMetric, Document.extra_metadata)
            .join(Document, Document.id == FinancialMetric.document_id)
            .where(and_(*conditions))
            .order_by(
                FinancialMetric.company_ticker,
                FinancialMetric.fiscal_year,
                FinancialMetric.fiscal_quarter,
            )
        )

        result = await db.execute(stmt)
        rows = result.all()

        records: list[MetricRecord] = []
        for metric, document_metadata in rows:
            metadata = document_metadata if isinstance(document_metadata, dict) else {}
            provenance = metadata.get("metric_provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            metric_provenance = provenance.get(metric.metric_name)
            metric_provenance = (
                metric_provenance if isinstance(metric_provenance, dict) else {}
            )
            accessions = metric_provenance.get("source_accessions", [])
            urls = metric_provenance.get("source_urls", [])
            records.append(
                MetricRecord(
                    company_ticker=metric.company_ticker or "",
                    fiscal_year=metric.fiscal_year,
                    fiscal_quarter=metric.fiscal_quarter,
                    period_end_date=metric.period_end_date,
                    metric_name=metric.metric_name,
                    value=metric.value,
                    unit=metric.unit,
                    segment=metric.segment or "Total",
                    is_adjusted=metric.is_adjusted,
                    document_id=metric.document_id,
                    source_page=metric.source_page,
                    source_provider=metadata.get("source_provider"),
                    source_accessions=tuple(accessions)
                    if isinstance(accessions, list)
                    else (),
                    source_urls=tuple(urls) if isinstance(urls, list) else (),
                    source_note=metric_provenance.get("calculation"),
                )
            )

        source_conditions = [Document.company_ticker.in_(tickers)]
        if fiscal_year_min is not None:
            source_conditions.append(Document.fiscal_year >= fiscal_year_min)
        if fiscal_year_max is not None:
            source_conditions.append(Document.fiscal_year <= fiscal_year_max)
        if fiscal_quarter is not None:
            source_conditions.append(Document.fiscal_quarter == fiscal_quarter)
        source_result = await db.execute(
            select(
                Document.company_ticker,
                Document.fiscal_year,
                Document.fiscal_quarter,
                Document.extra_metadata,
            ).where(and_(*source_conditions))
        )
        sec_periods = {
            (ticker, fiscal_year, fiscal_quarter)
            for ticker, fiscal_year, fiscal_quarter, metadata in source_result.all()
            if ticker is not None
            and fiscal_year is not None
            and isinstance(metadata, dict)
            and metadata.get("source_provider") == "sec_companyfacts"
        }
        records = _prefer_sec_records(records, sec_periods)

        log.info(
            "metric_query_complete",
            tickers=tickers,
            metrics=metric_names,
            records=len(records),
            sec_preferred_periods=len(sec_periods),
        )
        return records

    async def list_available_metrics(
        self,
        db: AsyncSession,
        ticker: Optional[str] = None,
    ) -> list[str]:
        """Return distinct metric names for a company (or all companies)."""
        from sqlalchemy import distinct

        stmt = select(distinct(FinancialMetric.metric_name))
        if ticker:
            stmt = stmt.where(FinancialMetric.company_ticker == ticker)
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def list_available_periods(
        self,
        db: AsyncSession,
        ticker: str,
        metric_name: str,
    ) -> list[str]:
        """Return period labels for which a given metric exists."""
        stmt = (
            select(
                FinancialMetric.fiscal_year,
                FinancialMetric.fiscal_quarter,
            )
            .where(
                FinancialMetric.company_ticker == ticker,
                FinancialMetric.metric_name == metric_name,
            )
            .order_by(FinancialMetric.fiscal_year, FinancialMetric.fiscal_quarter)
        )
        result = await db.execute(stmt)
        rows = result.all()
        labels = []
        for fy, fq in rows:
            labels.append(f"FY{fy} Q{fq}" if fq else f"FY{fy}")
        return labels
