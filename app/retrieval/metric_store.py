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

    @property
    def period_label(self) -> str:
        if self.fiscal_quarter:
            return f"FY{self.fiscal_year} Q{self.fiscal_quarter}"
        return f"FY{self.fiscal_year}"


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
            select(FinancialMetric)
            .where(and_(*conditions))
            .order_by(
                FinancialMetric.company_ticker,
                FinancialMetric.fiscal_year,
                FinancialMetric.fiscal_quarter,
            )
        )

        result = await db.execute(stmt)
        rows = result.scalars().all()

        records = [
            MetricRecord(
                company_ticker=r.company_ticker or "",
                fiscal_year=r.fiscal_year,
                fiscal_quarter=r.fiscal_quarter,
                period_end_date=r.period_end_date,
                metric_name=r.metric_name,
                value=r.value,
                unit=r.unit,
                segment=r.segment or "Total",
                is_adjusted=r.is_adjusted,
                document_id=r.document_id,
                source_page=r.source_page,
            )
            for r in rows
        ]

        log.info(
            "metric_query_complete",
            tickers=tickers,
            metrics=metric_names,
            records=len(records),
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
