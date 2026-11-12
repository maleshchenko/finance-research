from scripts.seed_demo import _is_sec_sourced_document


def test_reseed_preserves_sec_sourced_documents():
    assert _is_sec_sourced_document(
        {"source_provider": "sec_companyfacts", "id": 42}
    )
    assert not _is_sec_sourced_document(
        {"source_provider": None, "id": 43}
    )
