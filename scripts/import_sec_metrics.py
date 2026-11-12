"""Import quarterly GAAP metrics from the SEC Company Facts API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Document,
    DocumentType,
    FinancialMetric,
    MetricCategory,
)
from app.ingestion.sec_companyfacts import (
    SEC_COMPANIES,
    SECQuarter,
    build_quarterly_metrics,
    fetch_companyfacts,
)


def _document_metadata(quarter: SECQuarter) -> dict:
    metric_provenance = {
        name: {
            "source_tags": list(metric.source_tags),
            "source_accessions": list(metric.source_accessions),
            "source_urls": [quarter.source_url],
            "calculation": metric.calculation,
        }
        for name, metric in quarter.metrics.items()
    }
    return {
        "source_provider": "sec_companyfacts",
        "source_name": "SEC Company Facts",
        "source_url": quarter.source_url,
        "source_accessions": list(quarter.source_accessions),
        "source_form": quarter.source_form,
        "period_start": quarter.period_start.isoformat(),
        "period_end": quarter.period_end.isoformat(),
        "metric_provenance": metric_provenance,
    }


async def store_sec_quarters(
    db: AsyncSession,
    quarters: list[SECQuarter],
) -> int:
    """Idempotently replace records previously imported for these SEC periods."""
    imported_metrics = 0
    for quarter in quarters:
        source_filename = (
            f"sec_companyfacts_{quarter.ticker.lower()}_"
            f"fy{quarter.fiscal_year}_q{quarter.fiscal_quarter}.json"
        )
        result = await db.execute(
            select(Document).where(Document.source_filename == source_filename)
        )
        document = result.scalar_one_or_none()
        if document is not None:
            metadata = document.extra_metadata or {}
            if metadata.get("source_provider") != "sec_companyfacts":
                raise ValueError(
                    f"Refusing to overwrite non-SEC document {source_filename}."
                )
            await db.execute(
                delete(FinancialMetric).where(
                    FinancialMetric.document_id == document.id
                )
            )
            document.company_ticker = quarter.ticker
            document.company_name = quarter.company_name
            document.document_type = DocumentType.QUARTERLY_REPORT
            document.fiscal_year = quarter.fiscal_year
            document.fiscal_quarter = quarter.fiscal_quarter
            document.period_end_date = datetime.combine(
                quarter.period_end, datetime.min.time()
            )
            document.extra_metadata = _document_metadata(quarter)
        else:
            document = Document(
                company_ticker=quarter.ticker,
                company_name=quarter.company_name,
                document_type=DocumentType.QUARTERLY_REPORT,
                fiscal_year=quarter.fiscal_year,
                fiscal_quarter=quarter.fiscal_quarter,
                period_end_date=datetime.combine(
                    quarter.period_end, datetime.min.time()
                ),
                source_filename=source_filename,
                extra_metadata=_document_metadata(quarter),
            )
            db.add(document)
            await db.flush()

        db.add_all(
            FinancialMetric(
                document_id=document.id,
                company_ticker=quarter.ticker,
                fiscal_year=quarter.fiscal_year,
                fiscal_quarter=quarter.fiscal_quarter,
                period_end_date=datetime.combine(
                    quarter.period_end, datetime.min.time()
                ),
                metric_name=metric_name,
                metric_category=MetricCategory.INCOME_STATEMENT,
                value=metric.value,
                unit=metric.unit,
                segment="Total",
                is_adjusted=False,
                source_page=None,
            )
            for metric_name, metric in quarter.metrics.items()
        )
        imported_metrics += len(quarter.metrics)

    await db.flush()
    return imported_metrics


async def _build_records(
    tickers: list[str],
    fiscal_year: int,
    user_agent: str,
) -> list[SECQuarter]:
    records: list[SECQuarter] = []
    for index, ticker in enumerate(tickers):
        payload = await fetch_companyfacts(ticker, user_agent)
        records.extend(build_quarterly_metrics(payload, ticker, fiscal_year))
        if index < len(tickers) - 1:
            await asyncio.sleep(0.12)
    return records


def _summary(records: list[SECQuarter]) -> list[dict]:
    return [
        {
            "ticker": quarter.ticker,
            "fiscal_year": quarter.fiscal_year,
            "fiscal_quarter": quarter.fiscal_quarter,
            "period_end": quarter.period_end.isoformat(),
            "source_form": quarter.source_form,
            "source_accessions": list(quarter.source_accessions),
            "metrics": {
                name: {"value": metric.value, "unit": metric.unit}
                for name, metric in quarter.metrics.items()
            },
        }
        for quarter in records
    ]


async def _run(args: argparse.Namespace) -> None:
    records = await _build_records(args.tickers, args.fiscal_year, args.user_agent)
    if args.dry_run:
        print(json.dumps(_summary(records), indent=2))
        return

    from app.database.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        async with db.begin():
            count = await store_sec_quarters(db, records)
    print(
        f"Imported {count} SEC XBRL metrics for "
        f"{len(args.tickers)} tickers, FY{args.fiscal_year}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="+",
        choices=sorted(SEC_COMPANIES),
        default=["AAPL", "MSFT", "GOOGL"],
    )
    parser.add_argument("--fiscal-year", type=int, default=2023)
    parser.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT", ""),
        help="Descriptive SEC User-Agent including a contact address.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse facts without writing to PostgreSQL.",
    )
    args = parser.parse_args()
    if not args.user_agent.strip():
        parser.error(
            "Set SEC_USER_AGENT or pass --user-agent with an application name and contact."
        )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
