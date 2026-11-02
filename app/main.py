"""
FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000

Interactive docs at: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager

import asyncio

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sqlalchemy import select, func

from app.api.routes import documents, ingest, research
from app.config import settings
from app.database.session import engine, AsyncSessionLocal


# ── Logging ───────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    In production (no TTY) emit newline-delimited JSON so CloudWatch/Datadog
    can parse each log line as a structured event.
    In development emit human-readable coloured output.
    """
    is_tty = sys.stderr.isatty()

    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if is_tty:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


_configure_logging()
log = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify DB connectivity. Shutdown: dispose connection pool."""
    log.info(
        "app_starting",
        provider=settings.llm_provider,
        model=settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model,
    )

    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("database_ok")

    yield

    log.info("app_stopping")
    await engine.dispose()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Research Assistant",
    description=(
        "A hybrid RAG + structured-data agent for financial report analysis. "
        "Combines semantic retrieval, SQL-backed metric queries, and LLM synthesis "
        "to answer questions about earnings reports with full citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

# CORS: read allowed origins from env so prod and dev can differ
_cors_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


_REQUEST_TIMEOUT = 480


@app.middleware("http")
async def timing_and_timeout(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await asyncio.wait_for(call_next(request), timeout=_REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        elapsed = round((time.perf_counter() - t0) * 1000)
        log.warning("request_timeout", path=request.url.path, ms=elapsed)
        return JSONResponse(
            status_code=504,
            content={"detail": f"Request timed out after {_REQUEST_TIMEOUT}s."},
        )
    elapsed = round((time.perf_counter() - t0) * 1000)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=elapsed,
    )
    return response


# ── Error handling ────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=str(request.url), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(research.router)
app.include_router(documents.router)


@app.get("/health", tags=["health"])
async def health():
    """Liveness probe — used by ALB and Docker healthcheck."""
    return {"status": "ok"}


@app.get("/health/coverage", tags=["health"])
async def coverage():
    """
    Return which ticker × fiscal_year × fiscal_quarter combinations are
    in the database, plus per-ticker metric counts.

    Use this to spot data gaps before running research queries.
    """
    from app.database.models import Document, FinancialMetric

    async with AsyncSessionLocal() as db:
        # Documents per ticker/period
        doc_rows = (await db.execute(
            select(
                Document.company_ticker,
                Document.company_name,
                Document.fiscal_year,
                Document.fiscal_quarter,
                Document.document_type,
            ).order_by(
                Document.company_ticker,
                Document.fiscal_year,
                Document.fiscal_quarter,
            )
        )).all()

        # Metric counts per ticker
        metric_rows = (await db.execute(
            select(
                FinancialMetric.company_ticker,
                FinancialMetric.metric_name,
                func.count().label("count"),
            ).group_by(
                FinancialMetric.company_ticker,
                FinancialMetric.metric_name,
            ).order_by(
                FinancialMetric.company_ticker,
                FinancialMetric.metric_name,
            )
        )).all()

    documents = [
        {
            "ticker": r.company_ticker,
            "company": r.company_name,
            "fiscal_year": r.fiscal_year,
            "fiscal_quarter": r.fiscal_quarter,
            "document_type": r.document_type,
        }
        for r in doc_rows
    ]

    metrics: dict[str, dict[str, int]] = {}
    for r in metric_rows:
        metrics.setdefault(r.company_ticker, {})[r.metric_name] = r.count

    tickers = sorted({r["ticker"] for r in documents if r["ticker"]})
    gaps = []
    for ticker in tickers:
        ticker_docs = [d for d in documents if d["ticker"] == ticker]
        periods = {(d["fiscal_year"], d["fiscal_quarter"]) for d in ticker_docs}
        ticker_metrics = metrics.get(ticker, {})
        if not ticker_metrics:
            gaps.append(f"{ticker}: no structured metrics extracted")

    return {
        "documents": documents,
        "metrics_by_ticker": metrics,
        "tickers": tickers,
        "total_documents": len(documents),
        "gaps": gaps,
    }
