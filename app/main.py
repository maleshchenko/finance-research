"""
FastAPI application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000

Interactive docs at: http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import documents, ingest, research
from app.database.session import engine

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks on boot."""
    log.info("app_starting")

    # Verify database connectivity and pgvector extension
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("database_ok")

    yield

    log.info("app_stopping")
    await engine.dispose()


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

# CORS — tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=str(request.url), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── Route registration ────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(research.router)
app.include_router(documents.router)


@app.get("/health", tags=["health"])
async def health():
    """Liveness probe."""
    return {"status": "ok"}
