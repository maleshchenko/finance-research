"""Unit tests for the document chunker."""

from app.ingestion.chunker import TextChunk, _table_to_markdown, chunk_document
from app.ingestion.extractor import ExtractionResult, PageResult


def _make_result(pages: list[dict]) -> ExtractionResult:
    page_objs = [
        PageResult(
            page_number=p.get("page_number", i + 1),
            text=p.get("text", ""),
            section_heading=p.get("section_heading"),
            tables=p.get("tables", []),
        )
        for i, p in enumerate(pages)
    ]
    total_text = sum(len(p.text) for p in page_objs)
    return ExtractionResult(
        pages=page_objs,
        metrics=[],
        page_count=len(page_objs),
        raw_text_length=total_text,
    )


class TestTableToMarkdown:
    def test_basic_table(self):
        table = [["Metric", "Q3", "Q2"], ["Revenue", "85B", "90B"]]
        md = _table_to_markdown(table)
        assert "| Metric |" in md
        assert "| Revenue |" in md
        assert "---" in md

    def test_empty_table(self):
        assert _table_to_markdown([]) == ""

    def test_newlines_stripped_from_cells(self):
        table = [["Met\nric"], ["Val\nue"]]
        md = _table_to_markdown(table)
        assert "\n\n" not in md.split("---")[1]  # no double newlines in data rows


class TestChunkDocument:
    def test_produces_chunks(self):
        result = _make_result(
            [{"text": "Apple Inc. reported revenue of $85.8 billion for Q3 2024, " * 10}]
        )
        chunks = chunk_document(result)
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_indices_are_sequential(self):
        result = _make_result([{"text": "word " * 500}, {"text": "more words " * 500}])
        chunks = chunk_document(result)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_table_chunk_prefixed(self):
        table = [["Metric", "Value"], ["Revenue", "85,777"]]
        result = _make_result([{"text": "", "tables": [table]}])
        chunks = chunk_document(result)
        assert any("[TABLE" in c.content for c in chunks)

    def test_section_heading_propagated(self):
        result = _make_result(
            [{"text": "Some text about results.", "section_heading": "Results Of Operations"}]
        )
        chunks = chunk_document(result)
        assert all(c.section_heading == "Results Of Operations" for c in chunks)

    def test_empty_pages_produce_no_chunks(self):
        result = _make_result([{"text": "   "}, {"text": ""}])
        chunks = chunk_document(result)
        assert len(chunks) == 0
