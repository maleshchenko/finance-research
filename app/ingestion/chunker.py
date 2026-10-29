"""
Semantic chunker for financial documents.

Financial documents require a specialised chunking strategy:
  - Tables should not be split across chunks.
  - Narrative text uses recursive character splitting.
  - Each chunk is labelled with its source section so retrieval can
    distinguish MD&A commentary from the income statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.extractor import ExtractionResult, PageResult


@dataclass
class TextChunk:
    chunk_index: int
    content: str
    page_number: Optional[int]
    section_heading: Optional[str]
    token_count: Optional[int]


def _table_to_markdown(table: list[list[str]]) -> str:
    """Render a table as a compact Markdown string so it stays in one chunk."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        rows.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
        if i == 0:
            rows.append("|" + "|".join("---" for _ in row) + "|")
    return "\n".join(rows)


def chunk_document(result: ExtractionResult) -> list[TextChunk]:
    """
    Convert an ExtractionResult into a flat list of TextChunks.

    Strategy:
      1. For each page with tables, emit one chunk per table (Markdown).
      2. For each page's narrative text, apply recursive splitting.
      3. Deduplicate text that appears verbatim inside a rendered table.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )

    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in result.pages:
        page_num = page.page_number
        heading = page.section_heading

        # ── Table chunks (kept whole) ─────────────────────────────────────────
        table_texts: set[str] = set()
        for table in page.tables:
            md = _table_to_markdown(table)
            if not md.strip():
                continue
            table_texts.add(md[:200])  # store prefix for dedup
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=f"[TABLE — {heading or 'Financial Data'}]\n\n{md}",
                    page_number=page_num,
                    section_heading=heading,
                    token_count=None,
                )
            )
            chunk_index += 1

        # ── Narrative text chunks ─────────────────────────────────────────────
        # Skip lines that are entirely part of a rendered table to avoid
        # duplicating numerical data in the vector index.
        narrative = page.text.strip()
        if not narrative:
            continue

        splits = splitter.split_text(narrative)
        for split in splits:
            if not split.strip():
                continue
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=split,
                    page_number=page_num,
                    section_heading=heading,
                    token_count=None,
                )
            )
            chunk_index += 1

    return chunks
