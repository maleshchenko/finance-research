"""
Demo seed script, generates synthetic financial reports and ingests them.

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

import httpx

BASE_URL = "http://localhost:8000"

# Synthetic financial data, loosely inspired by public figures
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
    Build a valid PDF containing synthetic financial data.

    Uses stdlib (io.BytesIO).
    Text is written as hex strings inside the PDF content stream so there are
    no literal-string escaping issues whatsoever.
    Both pdfplumber and pdfminer can extract the text without any trouble.
    """
    data = COMPANIES[ticker]
    q_idx = quarter - 1
    rev = data["revenue"][q_idx]
    op = data["op_income"][q_idx]
    net = data["net_income"][q_idx]
    eps = data["eps"][q_idx]
    op_margin = round(op / rev * 100, 1)
    prev_rev = data["revenue"][q_idx - 1] if q_idx > 0 else rev * 0.9
    yoy_growth = round((rev - prev_rev) / prev_rev * 100, 1)
    fwd_rev = rev * random.uniform(1.03, 1.08)
    direction = "growth" if yoy_growth >= 0 else "a decline"

    lines = [
        f"{ticker} {data['name']}",
        f"Q{quarter} FY{year} Quarterly Report",
        "",
        "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
        "in millions except per share data",
        "",
        f"                   Q{quarter} FY{year}    Q{quarter} FY{year - 1}",
        f"Total Revenue      {rev:>12,.0f}    {int(rev * 0.88):>12,.0f}",
        f"Operating Income   {op:>12,.0f}    {int(op * 0.82):>12,.0f}",
        f"Operating Margin   {op_margin:>11.1f}%   {op_margin - 1.2:>11.1f}%",
        f"Net Income         {net:>12,.0f}    {int(net * 0.84):>12,.0f}",
        f"Diluted EPS        {eps:>12.2f}    {eps * 0.84:>12.2f}",
        "",
        "Management Discussion and Analysis",
        "",
        f"Revenue for Q{quarter} FY{year} was {rev / 1000:.1f} billion, representing",
        f"{direction} of {abs(yoy_growth):.1f}% year-over-year.",
        f"Operating income of {op / 1000:.1f} billion reflects an operating margin",
        f"of {op_margin:.1f}%, compared to {op_margin - 1.2:.1f}% in the prior-year period.",
        f"Net income was {net / 1000:.1f} billion with diluted EPS of {eps:.2f}.",
        "",
        "Forward-Looking Statements and Guidance",
        "",
        "For the upcoming quarter, management expects revenue in the range of",
        f"{fwd_rev * 0.98 / 1000:.1f} billion to {fwd_rev * 1.02 / 1000:.1f} billion.",
        f"Gross margin is expected to be approximately {op_margin - 2:.0f}% to {op_margin:.0f}%.",
        "This guidance reflects current expectations and is subject to risks.",
    ]

    # Encode each line as a PDF hex string: <hexbytes>
    # This completely sidesteps all literal-string escaping rules.
    def hex_str(text: str) -> bytes:
        return b"<" + text.encode("latin-1", errors="replace").hex().upper().encode() + b">"

    # PDF content stream: BT … ET with 14-point leading
    stream_cmds = [b"BT", b"/F1 10 Tf", b"50 750 Td", b"14 TL"]
    for line in lines:
        stream_cmds.append(hex_str(line) + b" Tj T*")
    stream_cmds.append(b"ET")
    stream = b"\n".join(stream_cmds)

    # Build the PDF byte-by-byte, tracking exact offsets for the xref table.
    buf = io.BytesIO()

    buf.write(b"%PDF-1.4\n")

    offsets: dict[int, int] = {}

    offsets[1] = buf.tell()
    buf.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    offsets[2] = buf.tell()
    buf.write(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

    offsets[3] = buf.tell()
    buf.write(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    offsets[4] = buf.tell()
    buf.write(b"4 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\nstream\n")
    buf.write(stream)
    buf.write(b"\nendstream\nendobj\n")

    offsets[5] = buf.tell()
    buf.write(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n")

    # Cross-reference table (each entry is exactly 20 bytes: 10+1+5+1+1+1+\n)
    xref_pos = buf.tell()
    buf.write(b"xref\n0 6\n")
    buf.write(b"0000000000 65535 f \n")  # object 0, always free
    for i in range(1, 6):
        buf.write(f"{offsets[i]:010d} 00000 n \n".encode())

    buf.write(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n")
    buf.write(str(xref_pos).encode() + b"\n%%EOF\n")

    return buf.getvalue()


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
