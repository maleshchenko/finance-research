from __future__ import annotations

from app.retrieval.vector_store import RetrievedChunk, reciprocal_rank_fusion


def _chunk(chunk_id: int, similarity: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        company_ticker="AAPL",
        company_name="Apple Inc.",
        document_type="quarterly_report",
        fiscal_year=2023,
        fiscal_quarter=1,
        page_number=1,
        section_heading="Guidance",
        content=f"chunk {chunk_id}",
        similarity_score=similarity,
    )


def test_reciprocal_rank_fusion_rewards_chunks_found_by_both_searches():
    semantic = [_chunk(1, 0.9), _chunk(2, 0.8)]
    lexical = [_chunk(2, None), _chunk(3, None)]

    results = reciprocal_rank_fusion(semantic, lexical, top_k=3)

    assert [chunk.chunk_id for chunk in results] == [2, 1, 3]
    assert results[0].similarity_score == 0.8
    assert results[0].retrieval_score > results[1].retrieval_score


def test_reciprocal_rank_fusion_preserves_lexical_only_results():
    lexical = [_chunk(7, None)]

    results = reciprocal_rank_fusion([], lexical, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == 7
    assert results[0].similarity_score is None
