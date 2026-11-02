"""
Semantic chunker for financial documents.

Financial documents require a specialised chunking strategy:
  - Tables should not be split across chunks.
  - Narrative text is split first at section boundaries, then by token count.
  - Each chunk is labelled with its source section so retrieval can
    distinguish MD&A commentary from the income statement.

Section-aware chunking (v2):
  The extractor already detects section headings per page. This module
  additionally scans *within* each page's narrative text for heading
  transitions (e.g. a new "Risk Factors" block starting mid-page) so that
  chunks never straddle two sections. Keeping section context tight
  improves retrieval precision for queries like "what guidance did
  management provide?" versus "what are the risk factors?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.extractor import ExtractionResult, PageResult, SECTION_HEADINGS

# Headings that signal the start of a new logical section within a page.
# We look for uppercase/title-case lines that look like known section titles.
_INLINE_HEADING_RE = re.compile(
    r"^(?:(?:Item\s+\d+[A-Za-z]?\.?\s*)?"
    r"(?:Management.s Discussion|Risk Factors|Liquidity|Capital Resources"
    r"|Results of Operations|Forward.Looking|Business Overview"
    r"|Critical Accounting|Quantitative.*Qualitative"
    r"|Condensed Consolidated|Notes to Financial"
    r"|Selected Financial Data|Executive Summary"
    r"|Guidance|Outlook|Operating Segment))",
    re.I | re.MULTILINE,
)


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


def _split_by_section(text: str, base_heading: Optional[str]) -> list[tuple[str, str]]:
    """
    Split `text` at inline section heading boundaries.

    Returns a list of (section_heading, text_segment) pairs.
    Each segment contains only text from that section.
    """
    segments: list[tuple[str, str]] = []
    current_heading = base_heading
    current_start = 0

    for match in _INLINE_HEADING_RE.finditer(text):
        start = match.start()
        if start > current_start:
            segment = text[current_start:start].strip()
            if segment:
                segments.append((current_heading or "General", segment))
        # The matched line is the new heading
        line_end = text.find("\n", start)
        if line_end == -1:
            line_end = len(text)
        current_heading = text[start:line_end].strip()
        current_start = line_end + 1

    # Remaining text after last heading
    tail = text[current_start:].strip()
    if tail:
        segments.append((current_heading or "General", tail))

    if not segments:
        # No headings found — return as single segment with original heading
        return [(base_heading or "General", text)]

    return segments


def chunk_document(result: ExtractionResult) -> list[TextChunk]:
    """
    Convert an ExtractionResult into a flat list of TextChunks.

    Strategy:
      1. For each page with tables, emit one chunk per table (Markdown).
      2. For each page's narrative text:
         a. Split at section heading boundaries within the page.
         b. Within each section segment, apply RecursiveCharacterTextSplitter.
      3. Each chunk carries the section_heading for metadata-aware retrieval.
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
        page_heading = page.section_heading  # heading detected by extractor for the whole page

        # ── Table chunks (kept whole) ─────────────────────────────────────────
        for table in page.tables:
            md = _table_to_markdown(table)
            if not md.strip():
                continue
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    content=f"[TABLE — {page_heading or 'Financial Data'}]\n\n{md}",
                    page_number=page_num,
                    section_heading=page_heading,
                    token_count=None,
                )
            )
            chunk_index += 1

        # ── Narrative text: section-aware then token-based splitting ──────────
        narrative = page.text.strip()
        if not narrative:
            continue

        # Split at section boundaries first
        section_segments = _split_by_section(narrative, page_heading)

        for section_heading, segment_text in section_segments:
            if not segment_text.strip():
                continue

            # Then split each section segment by token size
            splits = splitter.split_text(segment_text)
            for split in splits:
                if not split.strip():
                    continue
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_index,
                        content=split,
                        page_number=page_num,
                        section_heading=section_heading,
                        token_count=None,
                    )
                )
                chunk_index += 1

    return chunks
