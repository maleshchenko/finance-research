"""Metrics for labeled retrieval evaluations."""

from collections.abc import Sequence


def retrieval_metrics(
    retrieved_ids: Sequence[int],
    relevant_ids: set[int],
    *,
    k: int,
) -> dict[str, float]:
    """Calculate Recall@k and reciprocal rank for a labeled query."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    top_k_ids = retrieved_ids[:k]
    relevant_rank = next(
        (rank for rank, chunk_id in enumerate(top_k_ids, start=1) if chunk_id in relevant_ids),
        None,
    )
    return {
        "recall": len(set(top_k_ids) & relevant_ids) / len(relevant_ids),
        "reciprocal_rank": 1.0 / relevant_rank if relevant_rank else 0.0,
    }
