"""
LangChain tools surfaced to the LangGraph agent.

Five tools implement the hybrid retrieval strategy:

  1. retrieve_narrative  — semantic search over document chunks (RAG)
  2. query_metrics       — structured SQL query for numerical time series
  3. calculate           — Python calculations on returned metric series
  4. list_documents      — discover what's in the database
  5. list_metrics        — discover which metrics exist for a company

Each tool returns a JSON-serialisable dict so the LLM can inspect values
and the API can include them in the citation trail.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Optional

import structlog
from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentType
from app.retrieval.metric_store import MetricRecord, MetricStore
from app.retrieval.vector_store import RetrievalFilter, VectorRetriever

log = structlog.get_logger(__name__)

# Singletons — initialised once per worker process
_retriever = VectorRetriever()
_metric_store = MetricStore()


# ── Tool 1: Narrative retrieval ───────────────────────────────────────────────

def build_retrieve_narrative_tool(db: AsyncSession):
    """
    Factory that captures the current DB session in a closure.
    We recreate the tool per-request so it shares the request's session.
    """

    @tool
    async def retrieve_narrative(
        query: Annotated[str, "Natural-language search query"],
        tickers: Annotated[
            Optional[list[str]],
            "Filter to these company tickers, e.g. ['AAPL', 'MSFT']",
        ] = None,
        document_types: Annotated[
            Optional[list[str]],
            "Filter to document types: annual_report, quarterly_report, earnings_release",
        ] = None,
        fiscal_year_min: Annotated[Optional[int], "Earliest fiscal year"] = None,
        fiscal_year_max: Annotated[Optional[int], "Latest fiscal year"] = None,
        top_k: Annotated[int, "Max number of chunks to return"] = 6,
    ) -> str:
        """
        Retrieve narrative text from financial reports using semantic search.

        Use this for qualitative questions: management commentary, forward guidance,
        risk factors, strategy discussion, MD&A explanations.

        Returns a JSON list of chunks with source citations.
        """
        dtype_objs: list[DocumentType] | None = None
        if document_types:
            dtype_objs = [DocumentType(t) for t in document_types if t in DocumentType.__members__.values()]

        filters = RetrievalFilter(
            tickers=tickers,
            document_types=dtype_objs,
            fiscal_year_min=fiscal_year_min,
            fiscal_year_max=fiscal_year_max,
            top_k=top_k,
        )
        chunks = await _retriever.retrieve(db, query, filters)
        results = [
            {
                "citation": c.citation_label,
                "section": c.section_heading,
                "similarity": round(c.similarity_score, 3),
                "content": c.content,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]
        return json.dumps(results, ensure_ascii=False)

    return retrieve_narrative


# ── Tool 2: Structured metric query ──────────────────────────────────────────

def build_query_metrics_tool(db: AsyncSession):

    @tool
    async def query_metrics(
        tickers: Annotated[list[str], "Company tickers to query"],
        metric_names: Annotated[
            list[str],
            "Metric names, e.g. ['revenue', 'operating_margin', 'eps_diluted']",
        ],
        fiscal_year_min: Annotated[Optional[int], "Earliest fiscal year"] = None,
        fiscal_year_max: Annotated[Optional[int], "Latest fiscal year"] = None,
        fiscal_quarter: Annotated[
            Optional[int], "Specific quarter (1-4) or None for all"
        ] = None,
    ) -> str:
        """
        Query structured financial metrics from the database via SQL.

        Use this for numerical time-series questions:
          - "How did revenue change over 8 quarters?"
          - "Compare operating margins across AAPL, MSFT, GOOGL for FY2023"

        Returns a JSON list of metric records with period labels.
        """
        records = await _metric_store.query_metric_series(
            db,
            tickers=tickers,
            metric_names=metric_names,
            fiscal_year_min=fiscal_year_min,
            fiscal_year_max=fiscal_year_max,
            fiscal_quarter=fiscal_quarter,
        )
        results = [
            {
                "ticker": r.company_ticker,
                "period": r.period_label,
                "metric": r.metric_name,
                "value": r.value,
                "unit": r.unit,
                "segment": r.segment,
                "is_adjusted": r.is_adjusted,
                "document_id": r.document_id,
                "source_page": r.source_page,
            }
            for r in records
        ]
        return json.dumps(results, ensure_ascii=False)

    return query_metrics


# ── Tool 3: Python calculations ───────────────────────────────────────────────

@tool
def calculate(
    operation: Annotated[
        str,
        "One of: yoy_growth, qoq_growth, margin, cagr, mean, median, delta",
    ],
    values: Annotated[
        list[float],
        "Ordered series of values (oldest first for growth/cagr)",
    ],
    labels: Annotated[Optional[list[str]], "Period labels matching `values`"] = None,
) -> str:
    """
    Perform deterministic financial calculations on a series of metric values.

    Use this AFTER retrieving metrics to compute growth rates, margins, or averages.
    Returns a JSON object with the calculation results.
    """
    if not values:
        return json.dumps({"error": "empty values list"})

    labels = labels or [str(i) for i in range(len(values))]

    def safe_pct(new: float, old: float) -> Optional[float]:
        if old == 0:
            return None
        return round((new - old) / abs(old) * 100, 2)

    result: dict[str, Any] = {"operation": operation, "input_count": len(values)}

    if operation == "yoy_growth" and len(values) >= 2:
        result["changes"] = [
            {"from": labels[i - 1], "to": labels[i], "pct_change": safe_pct(values[i], values[i - 1])}
            for i in range(1, len(values))
        ]
    elif operation == "qoq_growth" and len(values) >= 2:
        result["changes"] = [
            {"from": labels[i - 1], "to": labels[i], "pct_change": safe_pct(values[i], values[i - 1])}
            for i in range(1, len(values))
        ]
    elif operation == "cagr" and len(values) >= 2:
        n = len(values) - 1
        if values[0] > 0:
            result["cagr_pct"] = round((math.pow(values[-1] / values[0], 1 / n) - 1) * 100, 2)
        else:
            result["cagr_pct"] = None
        result["periods"] = n
    elif operation == "margin":
        # values[0] = numerator (e.g. operating income), values[1] = denominator (revenue)
        if len(values) >= 2 and values[1] != 0:
            result["margin_pct"] = round(values[0] / values[1] * 100, 2)
        else:
            result["margin_pct"] = None
    elif operation == "mean":
        result["mean"] = round(sum(values) / len(values), 4)
        result["min"] = min(values)
        result["max"] = max(values)
    elif operation == "median":
        s = sorted(values)
        mid = len(s) // 2
        result["median"] = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    elif operation == "delta":
        result["absolute_change"] = round(values[-1] - values[0], 4)
        result["pct_change"] = safe_pct(values[-1], values[0])
        result["from_label"] = labels[0]
        result["to_label"] = labels[-1]
    else:
        result["error"] = f"Unsupported operation '{operation}'"

    return json.dumps(result)


# ── Tool 4: List documents ────────────────────────────────────────────────────

def build_list_documents_tool(db: AsyncSession):

    @tool
    async def list_documents(
        ticker: Annotated[Optional[str], "Filter by company ticker"] = None,
        document_type: Annotated[Optional[str], "Filter by document type"] = None,
    ) -> str:
        """
        List documents ingested into the knowledge base.

        Use this when the user asks what data is available, or to confirm
        that a specific company/period is present before running other tools.
        """
        from sqlalchemy import select
        from app.database.models import Document

        stmt = select(
            Document.id,
            Document.company_ticker,
            Document.company_name,
            Document.document_type,
            Document.fiscal_year,
            Document.fiscal_quarter,
            Document.period_end_date,
            Document.source_filename,
        )
        if ticker:
            stmt = stmt.where(Document.company_ticker == ticker)
        if document_type:
            stmt = stmt.where(Document.document_type == document_type)
        stmt = stmt.order_by(
            Document.company_ticker,
            Document.fiscal_year.desc(),
            Document.fiscal_quarter.desc().nullslast(),
        )

        result = await db.execute(stmt)
        docs = [
            {
                "id": row.id,
                "ticker": row.company_ticker,
                "company": row.company_name,
                "type": row.document_type,
                "fiscal_year": row.fiscal_year,
                "fiscal_quarter": row.fiscal_quarter,
                "period_end": row.period_end_date.isoformat() if row.period_end_date else None,
                "filename": row.source_filename,
            }
            for row in result.all()
        ]
        return json.dumps(docs, ensure_ascii=False)

    return list_documents


# ── Tool 5: List metrics ──────────────────────────────────────────────────────

def build_list_metrics_tool(db: AsyncSession):

    @tool
    async def list_metrics(
        ticker: Annotated[Optional[str], "Company ticker"] = None,
    ) -> str:
        """
        List available structured metric names in the database.

        Use this to discover what metrics have been extracted before
        calling query_metrics.
        """
        metrics = await _metric_store.list_available_metrics(db, ticker)
        return json.dumps({"available_metrics": metrics})

    return list_metrics


# ── Factory ───────────────────────────────────────────────────────────────────

def build_tools(db: AsyncSession) -> list:
    """Return all agent tools bound to the given DB session."""
    return [
        build_retrieve_narrative_tool(db),
        build_query_metrics_tool(db),
        calculate,
        build_list_documents_tool(db),
        build_list_metrics_tool(db),
    ]
