"""Research query API: ask questions, get cited answers."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import run_research_query
from app.api.schemas import ResearchRequest, ResearchResponse
from app.database.models import QueryLog
from app.database.session import get_db

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query", response_model=ResearchResponse)
async def query(
    body: ResearchRequest,
    db: AsyncSession = Depends(get_db),
) -> ResearchResponse:
    """
    Submit a research question and receive a cited answer.

    The agent will:
    1. Classify the question (narrative / numerical / hybrid)
    2. Retrieve relevant document chunks and/or structured metrics
    3. Perform any required calculations
    4. Synthesise a cited answer that distinguishes reported facts from interpretation

    Answers typically take 5–30 seconds depending on complexity.
    """
    t0 = time.perf_counter()

    result = await run_research_query(db=db, question=body.question)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Persist to audit log
    log_entry = QueryLog(
        question=body.question,
        classification=result.get("classification"),
        answer=result.get("answer"),
        citations=result.get("citations"),
        tools_used=result.get("tool_calls_made"),
        latency_ms=elapsed_ms,
        error=result.get("error"),
    )
    db.add(log_entry)
    await db.flush()

    return ResearchResponse(
        answer=result.get("answer") or "No answer generated.",
        classification=result.get("classification"),
        citations=result.get("citations", []),
        tool_calls_made=result.get("tool_calls_made", []),
        elapsed_seconds=result.get("elapsed_seconds", 0),
        error=result.get("error"),
    )
