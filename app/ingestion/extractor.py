"""
Document extraction layer.

Responsibilities:
  1. Parse PDF pages with pdfplumber (text + table detection).
  2. Detect financial tables and extract numerical metrics.
  3. Identify section headings (MD&A, Risk Factors, …) for metadata-aware chunking.
  4. Return a structured `ExtractionResult` ready for the chunker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
import structlog

log = structlog.get_logger(__name__)

# ── Canonical metric name mapping ─────────────────────────────────────────────
#
# Maps lowercased header tokens to (canonical_name, MetricCategory, unit).
# Extend this map to support more filings.

METRIC_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"total\s+revenue", re.I),      "revenue",          "income_statement", "USD_millions"),
    (re.compile(r"net\s+revenue", re.I),         "revenue",          "income_statement", "USD_millions"),
    (re.compile(r"gross\s+profit", re.I),        "gross_profit",     "income_statement", "USD_millions"),
    (re.compile(r"gross\s+margin", re.I),        "gross_margin",     "income_statement", "percent"),
    (re.compile(r"operating\s+income", re.I),    "operating_income", "income_statement", "USD_millions"),
    (re.compile(r"operating\s+margin", re.I),    "operating_margin", "income_statement", "percent"),
    (re.compile(r"ebitda", re.I),                "ebitda",           "income_statement", "USD_millions"),
    (re.compile(r"net\s+income", re.I),          "net_income",       "income_statement", "USD_millions"),
    (re.compile(r"earnings?\s+per\s+share", re.I), "eps_diluted",    "income_statement", "USD"),
    (re.compile(r"diluted\s+eps", re.I),         "eps_diluted",      "income_statement", "USD"),
    (re.compile(r"free\s+cash\s+flow", re.I),    "free_cash_flow",   "cash_flow",        "USD_millions"),
    (re.compile(r"capital\s+expenditure", re.I), "capex",            "cash_flow",        "USD_millions"),
    (re.compile(r"total\s+assets", re.I),        "total_assets",     "balance_sheet",    "USD_millions"),
    (re.compile(r"total\s+debt", re.I),          "total_debt",       "balance_sheet",    "USD_millions"),
    (re.compile(r"cash\s+and\s+equivalents", re.I), "cash",          "balance_sheet",    "USD_millions"),
]

SECTION_HEADINGS = re.compile(
    r"(?:^|\n)"
    r"(?:item\s+\d+[a-z]?\.?\s*)?"
    r"(management.s discussion|risk factors|liquidity|capital resources"
    r"|results of operations|forward.looking|business overview"
    r"|critical accounting|quantitative.*qualitative)",
    re.I,
)


@dataclass
class RawMetric:
    metric_name: str
    metric_category: str
    value: float
    unit: str
    page_number: int
    segment: str = "Total"
    is_adjusted: bool = False


@dataclass
class PageResult:
    page_number: int
    text: str
    section_heading: Optional[str] = None
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class ExtractionResult:
    pages: list[PageResult]
    metrics: list[RawMetric]
    page_count: int
    raw_text_length: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_numeric(raw: str) -> Optional[float]:
    """Convert a cell like '(1,234.5)' or '85,777' to a float.  Returns None if unparseable."""
    if not raw:
        return None
    s = raw.strip().replace(",", "").replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # Strip common suffixes: B M K
    scale = 1.0
    if s.endswith("B"):
        scale, s = 1_000.0, s[:-1]    # report as millions
    elif s.endswith("M"):
        scale, s = 1.0, s[:-1]
    elif s.endswith("K"):
        scale, s = 0.001, s[:-1]
    # Strip trailing % symbol → store as raw value (e.g. 28.4 for 28.4%)
    s = s.rstrip("%")
    try:
        value = float(s) * scale
        return -value if negative else value
    except ValueError:
        return None


def _match_metric(header: str) -> Optional[tuple[str, str, str]]:
    """Return (canonical_name, category, unit) for a recognised header, else None."""
    for pattern, name, category, unit in METRIC_PATTERNS:
        if pattern.search(header):
            return name, category, unit
    return None


def _extract_metrics_from_table(
    table: list[list[str | None]],
    page_number: int,
) -> list[RawMetric]:
    """
    Heuristic: if a table row header matches a known metric name and subsequent
    columns contain numeric values, emit one RawMetric per numeric cell.

    We only emit the *first* numeric column to avoid duplicate quarterly/YTD values.
    """
    metrics: list[RawMetric] = []
    if not table or len(table) < 2:
        return metrics

    for row in table:
        if not row:
            continue
        cells = [str(c).strip() if c else "" for c in row]
        header = cells[0]
        match = _match_metric(header)
        if match is None:
            continue
        name, category, unit = match
        is_pct = "percent" in unit

        for cell in cells[1:]:
            v = _parse_numeric(cell)
            if v is not None:
                # Heuristic: if value looks like a percentage expressed as a decimal (0.284),
                # convert to percent points.
                if is_pct and -1.0 <= v <= 1.0:
                    v = round(v * 100, 2)
                metrics.append(
                    RawMetric(
                        metric_name=name,
                        metric_category=category,
                        value=v,
                        unit=unit,
                        page_number=page_number,
                    )
                )
                break  # first column only

    return metrics


def _detect_heading(text: str) -> Optional[str]:
    m = SECTION_HEADINGS.search(text)
    if m:
        return m.group(1).strip().title()
    return None


# ── Public interface ──────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> ExtractionResult:
    """
    Extract text, detect sections, and pull structured metrics from tables.

    Uses pdfplumber's table detector for precise table extraction and falls back
    to the raw text when no table is found on a page.
    """
    log.info("extracting_pdf", path=str(path))

    pages: list[PageResult] = []
    all_metrics: list[RawMetric] = []
    current_heading: Optional[str] = None
    total_chars = 0

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number

            # ── Text ──────────────────────────────────────────────
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            total_chars += len(text)

            # Update running section heading
            heading = _detect_heading(text)
            if heading:
                current_heading = heading

            # ── Tables ────────────────────────────────────────────
            raw_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 4,
                }
            ) or []
            str_tables = [
                [[str(c) if c else "" for c in row] for row in tbl]
                for tbl in raw_tables
            ]

            # ── Metric extraction ─────────────────────────────────
            for tbl in str_tables:
                metrics = _extract_metrics_from_table(tbl, page_num)
                all_metrics.extend(metrics)

            pages.append(
                PageResult(
                    page_number=page_num,
                    text=text,
                    section_heading=current_heading,
                    tables=str_tables,
                )
            )

    log.info(
        "extraction_complete",
        pages=len(pages),
        metrics_found=len(all_metrics),
        chars=total_chars,
    )
    return ExtractionResult(
        pages=pages,
        metrics=all_metrics,
        page_count=len(pages),
        raw_text_length=total_chars,
    )
