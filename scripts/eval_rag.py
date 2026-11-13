"""Evaluate narrative retrieval against labeled evidence from the demo corpus.

Run after starting PostgreSQL, applying migrations, and ingesting the demo data:
    python -m scripts.eval_rag
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document, DocumentChunk
from app.database.session import AsyncSessionLocal, engine
from app.retrieval.evaluation import retrieval_metrics
from app.retrieval.vector_store import RetrievalFilter, VectorRetriever


class BenchmarkCase(TypedDict):
    id: str
    question: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    evidence_phrases: list[str]


BENCHMARK_CASES: list[BenchmarkCase] = [
    {
        "id": "aapl-q1-guidance",
        "question": "What revenue range did Apple's management give for the upcoming quarter?",
        "ticker": "AAPL",
        "fiscal_year": 2023,
        "fiscal_quarter": 1,
        "evidence_phrases": ["management expects revenue in the range"],
    },
    {
        "id": "msft-q4-guidance",
        "question": "What revenue guidance did Microsoft give for its upcoming quarter?",
        "ticker": "MSFT",
        "fiscal_year": 2023,
        "fiscal_quarter": 4,
        "evidence_phrases": ["management expects revenue in the range"],
    },
    {
        "id": "aapl-q1-revenue-commentary",
        "question": "How did Apple describe its Q1 FY2023 revenue performance?",
        "ticker": "AAPL",
        "fiscal_year": 2023,
        "fiscal_quarter": 1,
        "evidence_phrases": ["revenue for q1 fy2023 was"],
    },
]


async def _relevant_chunk_ids(db: AsyncSession, case: BenchmarkCase) -> set[int]:
    conditions = [
        Document.company_ticker == case["ticker"],
        Document.fiscal_year == case["fiscal_year"],
        Document.fiscal_quarter == case["fiscal_quarter"],
        *(DocumentChunk.content.ilike(f"%{phrase}%") for phrase in case["evidence_phrases"]),
    ]
    result = await db.execute(
        select(DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(*conditions)
    )
    return {row.id for row in result.all()}


async def run_benchmark(k: int) -> int:
    retriever = VectorRetriever()
    query_metrics: list[dict[str, float]] = []

    async with AsyncSessionLocal() as db:
        for case in BENCHMARK_CASES:
            relevant_ids = await _relevant_chunk_ids(db, case)
            if not relevant_ids:
                print(f"SKIP {case['id']}: no labeled evidence found in the database")
                continue

            chunks = await retriever.retrieve(
                db,
                case["question"],
                RetrievalFilter(
                    tickers=[case["ticker"]],
                    fiscal_year_min=case["fiscal_year"],
                    fiscal_year_max=case["fiscal_year"],
                    fiscal_quarter=case["fiscal_quarter"],
                    top_k=k,
                ),
            )
            metrics = retrieval_metrics(
                [chunk.chunk_id for chunk in chunks],
                relevant_ids,
                k=k,
            )
            query_metrics.append(metrics)
            print(
                f"{case['id']}: Recall@{k}={metrics['recall']:.3f} "
                f"MRR@{k}={metrics['reciprocal_rank']:.3f} "
                f"({len(relevant_ids)} labeled chunks)"
            )

    if not query_metrics:
        print("No benchmark cases could be evaluated.")
        return 1

    mean_recall = sum(item["recall"] for item in query_metrics) / len(query_metrics)
    mean_mrr = sum(item["reciprocal_rank"] for item in query_metrics) / len(query_metrics)
    print(f"Macro average: Recall@{k}={mean_recall:.3f} MRR@{k}={mean_mrr:.3f}")
    return 0


async def _main(k: int) -> int:
    try:
        return await run_benchmark(k)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the demo filing corpus.")
    parser.add_argument("--k", type=int, default=8, help="Cutoff used for Recall@k and MRR@k.")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    raise SystemExit(asyncio.run(_main(args.k)))
