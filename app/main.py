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

from app.api.routes import documents, ingest, research
from app.config import settings
from app.database.session import engine


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
    log.info("app_starting", model=settings.openai_model)

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


_REQUEST_TIMEOUT = 120  # seconds — matches ALB idle timeout


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
