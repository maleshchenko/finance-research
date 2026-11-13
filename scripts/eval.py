"""
Eval harness for the financial research assistant.

Runs a fixed set of known Q&A pairs against the running API and checks:
  - that required keywords appear in the answer
  - that expected tool calls were made
  - that the answer is not fabricated (no answer for missing data)

Usage:
    # Requires API running: uvicorn app.main:app
    python scripts/eval.py
    python scripts/eval.py --base-url http://localhost:8000 --verbose
    python scripts/eval.py --fail-fast

Exit code: 0 if all pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

BASE_URL = "http://localhost:8000"

# ── Test cases ────────────────────────────────────────────────────────────────
# Each case specifies:
#   question        - the research question
#   keywords        - ALL of these must appear in the answer (case-insensitive)
#   any_keywords    - at least ONE of these must appear (case-insensitive)
#   tools_used      - ALL of these must appear in tool_calls_made
#   no_fabricate    - if True, check that answer doesn't claim data is present when it shouldn't be
#   skip_if_missing - tickers that must be in the KB, skip the test if they're not

EVAL_CASES = [
    # ── Single-company narrative ───────────────────────────────────────────
    {
        "id": "aapl_revenue_q1",
        "question": "What was Apple's total revenue in Q1 FY2023?",
        "keywords": ["AAPL", "94"],          # $94.8B
        "any_keywords": ["revenue", "Revenue"],
        "tools_used": ["retrieve_narrative", "query_metrics"],
        "skip_if_missing": ["AAPL"],
    },
    {
        "id": "msft_eps_q2",
        "question": "What was Microsoft's diluted EPS in Q2 FY2023?",
        "keywords": ["MSFT", "2.45"],
        "any_keywords": ["EPS", "earnings per share", "diluted"],
        "tools_used": ["retrieve_narrative", "query_metrics"],
        "skip_if_missing": ["MSFT"],
    },
    {
        "id": "googl_op_margin",
        "question": "What was Alphabet's operating margin in Q3 FY2023?",
        "keywords": ["GOOGL"],
        "any_keywords": ["margin", "operating", "%"],
        "tools_used": ["retrieve_narrative", "query_metrics"],
        "skip_if_missing": ["GOOGL"],
    },

    # ── Multi-company comparison ───────────────────────────────────────────
    {
        "id": "compare_margins_all",
        "question": "Compare operating margins for AAPL, MSFT, and GOOGL across all quarters of FY2023.",
        "keywords": ["AAPL", "MSFT", "GOOGL"],
        "any_keywords": ["margin", "operating", "Q1", "Q2", "Q3", "Q4"],
        "tools_used": ["query_metrics"],
        "skip_if_missing": ["AAPL", "MSFT", "GOOGL"],
    },
    {
        "id": "compare_revenue_aapl_msft",
        "question": "Compare total revenue between Apple and Microsoft for Q4 FY2023.",
        "keywords": ["AAPL", "MSFT"],
        "any_keywords": ["revenue", "Revenue", "117", "62"],   # ~$117B AAPL, ~$62B MSFT
        "tools_used": ["retrieve_narrative", "query_metrics"],
        "skip_if_missing": ["AAPL", "MSFT"],
    },

    # ── Forward guidance ──────────────────────────────────────────────────
    {
        "id": "guidance_mention",
        "question": "What revenue guidance did Apple management provide in Q1 FY2023?",
        "keywords": ["AAPL"],
        "any_keywords": ["guidance", "expects", "billion", "outlook", "management"],
        "tools_used": ["retrieve_narrative"],
        "skip_if_missing": ["AAPL"],
    },

    # ── Discovery ──────────────────────────────────────────────────────────
    {
        "id": "what_data_available",
        "question": "What companies and periods are available in the knowledge base?",
        "keywords": [],
        "any_keywords": ["AAPL", "MSFT", "GOOGL", "Apple", "Microsoft", "Alphabet"],
        "tools_used": [],
        "skip_if_missing": [],
    },

    # ── Missing data integrity ─────────────────────────────────────────────
    {
        "id": "unknown_company",
        "question": "What was XYZ Fictional Corp's revenue in Q1 FY2023?",
        "keywords": [],
        "any_keywords": ["not available", "not found", "no data", "knowledge base", "don't have"],
        "tools_used": [],
        "skip_if_missing": [],
        "no_fabricate": True,
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    passed: bool
    skipped: bool = False
    skip_reason: str = ""
    failures: list[str] = field(default_factory=list)
    answer_preview: str = ""
    tool_calls_made: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


async def _get_known_tickers(client: httpx.AsyncClient) -> set[str]:
    try:
        resp = await client.get("/documents/")
        resp.raise_for_status()
        docs = resp.json()
        return {d["company_ticker"] for d in docs if d.get("company_ticker")}
    except Exception:
        return set()


async def run_case(
    client: httpx.AsyncClient,
    case: dict,
    known_tickers: set[str],
    verbose: bool = False,
) -> EvalResult:
    case_id = case["id"]

    # Skip if required tickers are missing from KB
    missing = [t for t in case.get("skip_if_missing", []) if t not in known_tickers]
    if missing:
        return EvalResult(
            case_id=case_id,
            passed=True,
            skipped=True,
            skip_reason=f"tickers not in KB: {missing}",
        )

    t0 = time.perf_counter()
    try:
        resp = await client.post(
            "/research/query",
            json={"question": case["question"]},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return EvalResult(
            case_id=case_id,
            passed=False,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    answer = (data.get("answer") or "").lower()
    tool_calls = data.get("tool_calls_made", [])
    evidence_sources = data.get("evidence_sources", [])
    failures = []

    # Check required keywords
    for kw in case.get("keywords", []):
        if kw.lower() not in answer:
            failures.append(f"missing keyword: '{kw}'")

    # Check any_keywords (at least one must appear)
    any_kws = case.get("any_keywords", [])
    if any_kws and not any(kw.lower() in answer for kw in any_kws):
        failures.append(f"none of any_keywords found: {any_kws}")

    # Check required tool calls
    for tool in case.get("tools_used", []):
        if tool not in tool_calls:
            failures.append(f"expected tool not used: '{tool}'")

    # No-fabricate check: answer should signal missing data
    if case.get("no_fabricate") and data.get("error") is None:
        fabrication_signals = [
            "not available", "not found", "no data", "knowledge base",
            "don't have", "cannot find", "no information", "not in",
        ]
        if not any(sig in answer for sig in fabrication_signals):
            failures.append("possible fabrication: no 'data not available' signal for unknown entity")

    result = EvalResult(
        case_id=case_id,
        passed=len(failures) == 0,
        failures=failures,
        answer_preview=data.get("answer", "")[:200],
        tool_calls_made=tool_calls,
        evidence_sources=evidence_sources,
        elapsed_seconds=round(elapsed, 1),
        error=data.get("error"),
    )

    if verbose:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n  [{status}] {case_id} ({elapsed:.1f}s)")
        print(f"    Q: {case['question'][:80]}")
        print(f"    tools: {tool_calls}")
        if evidence_sources:
            print(f"    sources: {evidence_sources[:3]}")
        if failures:
            for f in failures:
                print(f"    ✗ {f}")
        print(f"    A: {result.answer_preview[:120]}…")

    return result


async def main(base_url: str, verbose: bool, fail_fast: bool) -> int:
    print(f"\n🔬 Financial Research Eval Harness")
    print(f"   API: {base_url}")
    print(f"   Cases: {len(EVAL_CASES)}\n")

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        # Check API health
        try:
            resp = await client.get("/health")
            resp.raise_for_status()
        except Exception as exc:
            print(f"✗ API not reachable at {base_url}: {exc}")
            return 1

        known_tickers = await _get_known_tickers(client)
        print(f"   KB tickers: {sorted(known_tickers) or '(none)'}\n")

    results: list[EvalResult] = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        for case in EVAL_CASES:
            if not verbose:
                print(f"  Running {case['id']}…", end=" ", flush=True)
            result = await run_case(client, case, known_tickers, verbose=verbose)
            results.append(result)
            if not verbose:
                if result.skipped:
                    print(f"SKIP ({result.skip_reason})")
                elif result.passed:
                    print(f"PASS ({result.elapsed_seconds}s)")
                else:
                    print(f"FAIL — {'; '.join(result.failures)}")
            if fail_fast and not result.passed and not result.skipped:
                break

    # Summary
    passed = [r for r in results if r.passed and not r.skipped]
    failed = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]
    total_run = len(passed) + len(failed)

    print(f"\n{'─' * 50}")
    print(f"Results: {len(passed)}/{total_run} passed  |  {len(skipped)} skipped")

    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"  ✗ {r.case_id}: {'; '.join(r.failures)}")
            if r.error:
                print(f"    error: {r.error}")
    else:
        print("All cases passed! ✓")

    return 0 if not failed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval harness for the financial research API")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.base_url, args.verbose, args.fail_fast)))
