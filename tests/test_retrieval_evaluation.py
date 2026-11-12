import pytest

from app.retrieval.evaluation import retrieval_metrics


def test_retrieval_metrics_measure_recall_and_first_relevant_rank():
    metrics = retrieval_metrics(
        [10, 20, 30, 40],
        {20, 40, 50},
        k=3,
    )

    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["reciprocal_rank"] == pytest.approx(1 / 2)


def test_retrieval_metrics_returns_zero_when_no_relevant_chunk_is_retrieved():
    metrics = retrieval_metrics([10, 20], {30}, k=2)

    assert metrics == {"recall": 0.0, "reciprocal_rank": 0.0}


@pytest.mark.parametrize(
    ("retrieved_ids", "relevant_ids", "k"),
    [([], {1}, 0), ([], set(), 1)],
)
def test_retrieval_metrics_rejects_invalid_inputs(retrieved_ids, relevant_ids, k):
    with pytest.raises(ValueError):
        retrieval_metrics(retrieved_ids, relevant_ids, k=k)
