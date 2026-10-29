"""
Demo seed script — generates synthetic financial reports and ingests them.

This lets you run the assistant end-to-end without real SEC filings.

Usage:
    python scripts/seed_demo.py

It creates three synthetic PDF reports for AAPL, MSFT, and GOOGL covering
FY2023 Q1–Q4, then calls the ingestion API.
"""

from __future__ import annotations

import asyncio
import io
import random
from datetime import date

import httpx

BASE_URL = "http://localhost:8000"

# Synthetic financial data — loosely inspired by public figures
COMPANIES = {
    "AAPL": {
        "name": "Apple Inc.",
        "revenue": [94_836, 89_498, 81_797, 117_154],  # Q1–Q4 FY2023 (millions)
        "op_income": [30_328, 28_318, 23_134, 39_100],
        "net_income": [24_160, 20_721, 19_881, 29_959],
        "eps": [1.52, 1.29, 1.26, 1.88],
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "revenue": [52_747, 52_857, 56_189, 62_020],
        "op_income": [22_352, 22_352, 24_255, 27_032],
        "net_income": [18_299, 18_299, 20_081, 22_291],
        "eps": [2.45, 2.45, 2.69, 2.99],
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "revenue": [69_787, 69_787, 74_599, 86_310],
        "op_income": [21_343, 17_139, 21_838, 23_937],
        "net_income": [15_051, 18_368, 19_689, 20_402],
        "eps": [1.17, 1.44, 1.55, 1.64],
    },
}


def _make_fake_pdf(ticker: str, quarter: int, year: int = 2023) -> bytes:
    """
    Build a minimal but realistic-looking PDF text using reportlab-like structure.
    We use fpdf2 if available, otherwise create a raw minimal PDF.
    """
    try:
        from fpdf import FPDF

        data = COMPANIES[ticker]
        q_idx = quarter - 1
        rev = data["revenue"][q_idx]
        op = data["op_income"][q_idx]
        net = data["net_income"][q_idx]
        eps = data["eps"][q_idx]
        op_margin = round(op / rev * 100, 1)
        prev_rev = data["revenue"][q_idx - 1] if q_idx > 0 else rev * 0.9
        yoy_growth = round((rev - prev_rev) / prev_rev * 100, 1)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"{data['name']} ({ticker})", ln=True)
        pdf.cell(0, 10, f"Q{quarter} FY{year} Earnings Release", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Quarter ended: March 31, {year}", ln=True)
        pdf.ln(5)

        # Income Statement Table
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(80, 7, "")
        pdf.cell(40, 7, f"Q{quarter} FY{year}", align="R")
        pdf.cell(40, 7, f"Q{quarter} FY{year-1}", align="R", ln=True)

        rows = [
            ("Total Revenue", f"{rev:,.0f}", f"{int(rev*0.88):,.0f}"),
            ("Gross Profit", f"{int(rev*0.43):,.0f}", f"{int(rev*0.42):,.0f}"),
            ("Operating Income", f"{op:,.0f}", f"{int(op*0.82):,.0f}"),
            ("Operating Margin", f"{op_margin}%", f"{op_margin-1.2}%"),
            ("Net Income", f"{net:,.0f}", f"{int(net*0.84):,.0f}"),
            ("Diluted EPS", f"${eps:.2f}", f"${eps*0.84:.2f}"),
        ]
        for label, current, prior in rows:
            pdf.cell(80, 6, label)
            pdf.cell(40, 6, current, align="R")
            pdf.cell(40, 6, prior, align="R", ln=True)

        # MD&A section
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Management's Discussion and Analysis", ln=True)
        pdf.set_font("Helvetica", "", 10)
        commentary = (
            f"Revenue for Q{quarter} FY{year} was ${rev/1000:.1f} billion, representing "
            f"{'growth' if yoy_growth >= 0 else 'a decline'} of {abs(yoy_growth):.1f}% "
            f"year-over-year. Operating income of ${op/1000:.1f} billion reflects an "
            f"operating margin of {op_margin}%, compared to {op_margin - 1.2}% in the "
            f"prior-year period. The improvement was driven by continued operational "
            f"efficiency and favorable product mix. "
        )
        pdf.multi_cell(0, 6, commentary)

        # Forward guidance section
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Forward-Looking Statements and Guidance", ln=True)
        pdf.set_font("Helvetica", "", 10)
        fwd_rev = rev * random.uniform(1.03, 1.08)
        guidance = (
            f"For the upcoming quarter, management expects revenue in the range of "
            f"${fwd_rev*0.98/1000:.1f} billion to ${fwd_rev*1.02/1000:.1f} billion. "
            f"Gross margin is expected to be approximately {op_margin - 2:.0f}% to "
            f"{op_margin:.0f}%. This guidance reflects management's current expectations "
            f"and is subject to risks and uncertainties. "
        )
        pdf.multi_cell(0, 6, guidance)

        return bytes(pdf.output())

    except ImportError:
        # Minimal valid PDF without fpdf2
        content = f"""\
{ticker} {data['name']}
Q{quarter} FY{year} Quarterly Report

CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
(in millions, except per share data)

                        Q{quarter} FY{year}    Q{quarter} FY{year-1}
Total Revenue           {rev:>10,.0f}    {int(rev*0.88):>10,.0f}
Operating Income        {op:>10,.0f}    {int(op*0.82):>10,.0f}
Operating Margin        {op_margin:>9.1f}%   {op_margin-1.2:>9.1f}%
Net Income              {net:>10,.0f}    {int(net*0.84):>10,.0f}
Diluted EPS             {eps:>10.2f}    {eps*0.84:>10.2f}

Management's Discussion and Analysis

Revenue for Q{quarter} FY{year} was ${rev/1000:.1f} billion, representing
{'growth' if yoy_growth >= 0 else 'a decline'} of {abs(yoy_growth):.1f}% year-over-year.
Operating income of ${op/1000:.1f} billion reflects an operating margin
of {op_margin}%, compared to {op_margin - 1.2}% in the prior-year period.

Forward-Looking Statements and Guidance

For the upcoming quarter, management expects revenue in the range of
${fwd_rev*0.98/1000:.1f} billion to ${fwd_rev*1.02/1000:.1f} billion.
""".encode()
        # Wrap in minimal PDF structure
        objects = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
        stream = b"BT /F1 10 Tf 50 750 Td (" + content[:200].replace(b"\n", b" ") + b") Tj ET"
        objects.append(b"4 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream\nendobj\n")
        objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        offsets = []
        body = b"%PDF-1.4\n"
        for obj in objects:
            offsets.append(len(body))
            body += obj

        xref_offset = len(body)
        xref = b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
        for off in offsets:
            xref += f"{off:010d} 00000 n \n".encode()
        body += xref
        body += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"
        return body


async def seed():
    print("🌱 Seeding demo financial data...")
    print(f"   API: {BASE_URL}")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        # Check health
        resp = await client.get("/health")
        resp.raise_for_status()
        print("   ✓ API is healthy")

        for ticker, data in COMPANIES.items():
            for quarter in range(1, 5):
                print(f"   Ingesting {ticker} Q{quarter} FY2023...", end=" ")
                pdf_bytes = _make_fake_pdf(ticker, quarter)
                filename = f"{ticker}_Q{quarter}_FY2023.pdf"

                resp = await client.post(
                    "/ingest/upload",
                    files={"file": (filename, pdf_bytes, "application/pdf")},
                    params={
                        "ticker": ticker,
                        "company_name": data["name"],
                        "document_type": "quarterly_report",
                        "fiscal_year": 2023,
                        "fiscal_quarter": quarter,
                    },
                )
                if resp.status_code == 201:
                    r = resp.json()
                    print(f"✓ doc_id={r['document_id']} chunks={r['chunks_created']} metrics={r['metrics_created']}")
                else:
                    print(f"✗ HTTP {resp.status_code}: {resp.text[:200]}")

    print("\n✅ Seeding complete! Try these queries:")
    print('   curl -X POST http://localhost:8000/research/query \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"question": "Compare operating margins for AAPL, MSFT, and GOOGL across all four quarters of FY2023"}\'')


if __name__ == "__main__":
    asyncio.run(seed())
