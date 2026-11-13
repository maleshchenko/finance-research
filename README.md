# Financial Research Assistant

A research assistant that answers questions about financial reports, earnings releases, and economic data, citing every claim.

Built with **LangChain**, **LangGraph**, **PostgreSQL + pgvector**, and **FastAPI**.

## Why not pure RAG?

Semantic search retrieves relevant paragraphs but struggles to reconstruct a complete numerical series. This project uses a hybrid approach:

| Question type | Strategy |
|---|---|
| "What did management say about margins?" | Hybrid RAG — vector + PostgreSQL full-text search, fused by rank |
| "How did revenue change over 8 quarters?" | SQL query -> Python calculation |
| "Compare margins and explain the trend" | SQL metrics, then LLM interpretation |

Explicit operating-margin comparisons for a fiscal year use structured SQL
metrics and deterministic tables. Numeric-only questions skip LLM synthesis and
grounding review; adding "explain the trend" requests structured LLM trend grouping
that is checked against deterministic quarter calculations before being rendered.
The interpretation describes observed patterns, not causes; causal questions use
narrative retrieval.
SEC Company Facts records take precedence over demo/PDF metrics for covered
periods; citations include accession numbers and calculation notes.

Narrative retrieval combines pgvector similarity with PostgreSQL full-text
search using reciprocal-rank fusion. This helps recover both semantically
related passages and exact financial terms.

## Importing SEC metrics

The demo seed creates synthetic values. Import FY2023 SEC Company Facts before
asking for authoritative numbers. A dry run only previews the records:

```bash
docker compose exec \
  -e SEC_USER_AGENT="finance-research/0.1 your-name@example.com" \
  api python -m scripts.import_sec_metrics \
  --tickers AAPL MSFT GOOGL --fiscal-year 2023 --dry-run

# Import or refresh the SEC rows (omit --dry-run).
docker compose exec \
  -e SEC_USER_AGENT="finance-research/0.1 your-name@example.com" \
  api python -m scripts.import_sec_metrics \
  --tickers AAPL MSFT GOOGL --fiscal-year 2023
```

The dry run does not write to PostgreSQL. Re-running the importer updates its
SEC records. `python scripts/seed_demo.py --reseed` refreshes demo PDFs but
preserves SEC-sourced documents and their metrics.

## Stack

- **LangGraph** — state machine: classify → ReAct agent loop → synthesise
- **LangChain** — tool definitions, OpenAI embeddings, ReAct agent
- **PostgreSQL + pgvector** — vector chunks (HNSW index) + structured metrics in the same DB
- **FastAPI** — async REST API with Pydantic schemas
- **pdfplumber** — table-aware PDF extraction
- **Alembic** — database migrations

## Quick start

### With OpenAI

```bash
# 1. Copy env and add your OpenAI key
cp .env.example .env          # set OPENAI_API_KEY=sk-...

# 2. Start the stack (applies migrations on boot)
docker compose up --build

# 3. Seed 12 synthetic reports (AAPL, MSFT, GOOGL × 4 quarters)
python scripts/seed_demo.py

# 4. Ask a question
curl -s -X POST http://localhost:8000/research/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend."}' \
  | jq '.answer | fromjson'
```

### With Ollama (fully local, no API key)

```bash
# 1. Pull the models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Configure
cp .env.example .env
# Edit .env:
#   LLM_PROVIDER=ollama
#   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
#   VECTOR_DIMENSION=768       # nomic-embed-text produces 768-dim vectors

# 3. Start the stack (Ollama runs as a sidecar)
docker compose --profile ollama up --build

# 4. Same seed + query commands as above
```

### Output

```
curl -s -X POST http://localhost:8000/research/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare operating margins for AAPL, MSFT, and GOOGL across FY2023 and explain the trend."}' \
  | jq '{answer, grounding_status, unverified_draft, grounding_issues}'

{
  "summary": "Overall, operating margins increased for MSFT and GOOGL; decreased for AAPL.",
  "claims": [
    {
      "text": "AAPL operating margin for FY2023 Q1 is 30.7424%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q1 · doc:130 · SEC accession(s): 0000320193-23-000006](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)"
      ]
    },
    {
      "text": "AAPL operating margin for FY2023 Q2 is 29.86%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q2 · doc:131 · SEC accession(s): 0000320193-23-000064](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)"
      ]
    },
    {
      "text": "AAPL operating margin for FY2023 Q3 is 28.1159%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q3 · doc:132 · SEC accession(s): 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)"
      ]
    },
    {
      "text": "AAPL operating margin for FY2023 Q4 is 30.1336%. Q4 line items are FY amounts minus Q1–Q3; operating margin is calculated as Q4 operating income / Q4 revenue × 100.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q4 · doc:133 · SEC accession(s): 0000320193-23-000106, 0000320193-23-000006, 0000320193-23-000064, 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)"
      ]
    },
    {
      "text": "MSFT operating margin for FY2023 Q1 is 42.9312%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q1 · doc:134 · SEC accession(s): 0001564590-22-035087](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)"
      ]
    },
    {
      "text": "MSFT operating margin for FY2023 Q2 is 38.6733%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q2 · doc:135 · SEC accession(s): 0001564590-23-000733](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)"
      ]
    },
    {
      "text": "MSFT operating margin for FY2023 Q3 is 42.2877%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q3 · doc:136 · SEC accession(s): 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)"
      ]
    },
    {
      "text": "MSFT operating margin for FY2023 Q4 is 43.165%. Q4 line items are FY amounts minus Q1–Q3; operating margin is calculated as Q4 operating income / Q4 revenue × 100.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q4 · doc:137 · SEC accession(s): 0000950170-23-035122, 0001564590-22-035087, 0001564590-23-000733, 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)"
      ]
    },
    {
      "text": "GOOGL operating margin for FY2023 Q1 is 24.9545%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q1 · doc:138 · SEC accession(s): 0001652044-23-000045](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    },
    {
      "text": "GOOGL operating margin for FY2023 Q2 is 29.2719%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q2 · doc:139 · SEC accession(s): 0001652044-23-000070](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    },
    {
      "text": "GOOGL operating margin for FY2023 Q3 is 27.8291%. Operating margin = operating income / revenue × 100.",
      "classification": "REPORTED",
      "citations": [
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q3 · doc:140 · SEC accession(s): 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    },
    {
      "text": "GOOGL operating margin for FY2023 Q4 is 27.4557%. Q4 line items are FY amounts minus Q1–Q3; operating margin is calculated as Q4 operating income / Q4 revenue × 100.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q4 · doc:141 · SEC accession(s): 0001652044-24-000022, 0001652044-23-000045, 0001652044-23-000070, 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    },
    {
      "text": "AAPL operating margin decreased from FY2023 Q1 to FY2023 Q2; decreased from FY2023 Q2 to FY2023 Q3; increased from FY2023 Q3 to FY2023 Q4; overall decreased from FY2023 Q1 to FY2023 Q4.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q1 · doc:130 · SEC accession(s): 0000320193-23-000006](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q2 · doc:131 · SEC accession(s): 0000320193-23-000064](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q3 · doc:132 · SEC accession(s): 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q4 · doc:133 · SEC accession(s): 0000320193-23-000106, 0000320193-23-000006, 0000320193-23-000064, 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)"
      ]
    },
    {
      "text": "MSFT operating margin decreased from FY2023 Q1 to FY2023 Q2; increased from FY2023 Q2 to FY2023 Q3; increased from FY2023 Q3 to FY2023 Q4; overall increased from FY2023 Q1 to FY2023 Q4.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q1 · doc:134 · SEC accession(s): 0001564590-22-035087](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q2 · doc:135 · SEC accession(s): 0001564590-23-000733](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q3 · doc:136 · SEC accession(s): 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q4 · doc:137 · SEC accession(s): 0000950170-23-035122, 0001564590-22-035087, 0001564590-23-000733, 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)"
      ]
    },
    {
      "text": "GOOGL operating margin increased from FY2023 Q1 to FY2023 Q2; decreased from FY2023 Q2 to FY2023 Q3; decreased from FY2023 Q3 to FY2023 Q4; overall increased from FY2023 Q1 to FY2023 Q4.",
      "classification": "CALCULATED",
      "citations": [
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q1 · doc:138 · SEC accession(s): 0001652044-23-000045](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q2 · doc:139 · SEC accession(s): 0001652044-23-000070](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q3 · doc:140 · SEC accession(s): 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q4 · doc:141 · SEC accession(s): 0001652044-24-000022, 0001652044-23-000045, 0001652044-23-000070, 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    },
    {
      "text": "Overall, operating margins increased for MSFT and GOOGL; decreased for AAPL.",
      "classification": "INTERPRETATION",
      "citations": [
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q1 · doc:130 · SEC accession(s): 0000320193-23-000006](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q2 · doc:131 · SEC accession(s): 0000320193-23-000064](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q3 · doc:132 · SEC accession(s): 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: AAPL operating_margin · FY2023 Q4 · doc:133 · SEC accession(s): 0000320193-23-000106, 0000320193-23-000006, 0000320193-23-000064, 0000320193-23-000077](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q1 · doc:134 · SEC accession(s): 0001564590-22-035087](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q2 · doc:135 · SEC accession(s): 0001564590-23-000733](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q3 · doc:136 · SEC accession(s): 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: MSFT operating_margin · FY2023 Q4 · doc:137 · SEC accession(s): 0000950170-23-035122, 0001564590-22-035087, 0001564590-23-000733, 0000950170-23-014423](https://data.sec.gov/api/xbrl/companyfacts/CIK0000789019.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q1 · doc:138 · SEC accession(s): 0001652044-23-000045](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q2 · doc:139 · SEC accession(s): 0001652044-23-000070](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q3 · doc:140 · SEC accession(s): 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)",
        "[STRUCTURED DATA: GOOGL operating_margin · FY2023 Q4 · doc:141 · SEC accession(s): 0001652044-24-000022, 0001652044-23-000045, 0001652044-23-000070, 0001652044-23-000094](https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json)"
      ]
    }
  ],
  "limitations": []
}
```

> **Note on embedding dimensions:** the `VECTOR_DIMENSION` setting must match
> the embedding model and must be set **before** the first `alembic upgrade head`
> — pgvector columns are fixed-width. If you need to change it later, drop and
> recreate the database.

Interactive API docs: **http://localhost:8000/docs**

## JSON answer payload

The API response itself is JSON, and its `answer` field is a JSON-serialized string.
Parse that string with `jq '.answer | fromjson'` (or `json.loads(response["answer"])`
in Python). The parsed object always has:

- `summary`: concise answer summary
- `claims`: evidence-backed claims, each with a `text`, `classification`, and non-empty
  `citations` list
- `limitations`: data gaps and qualifications

The allowed claim classifications are `REPORTED`, `CALCULATED`, `INTERPRETATION`, and
`WEB SOURCE`. `unverified_draft`, when present, uses the same JSON format.

## Watching live logs

Follow API startup, request, retrieval, synthesis, and grounding-review events:

```bash
docker compose logs -f --tail=100 api
```

Each request gets a `request_id`, included in the structured log events and
returned in the `X-Request-ID` response header. Match events with the same ID to
follow a request through the agent. When running Uvicorn directly, the same
events are printed in the terminal:

```bash
uvicorn app.main:app --reload --port 8000
```

## Uploading a real PDF

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@AAPL_10Q_Q3_2024.pdf" \
  -F "ticker=AAPL" \
  -F "document_type=quarterly_report" \
  -F "fiscal_year=2024" \
  -F "fiscal_quarter=3"
```

## Project layout

```
app/
  ingestion/      PDF extraction, table-aware chunking, embedding pipeline
  retrieval/      hybrid text/vector search + typed SQL metric queries
  agent/          LangGraph graph, 5 LangChain tools
  api/routes/     POST /ingest/upload, POST /research/query, GET /documents/…
  database/       SQLAlchemy models, Alembic migrations
scripts/
  seed_demo.py    Generate and ingest synthetic reports
  eval_rag.py     Measure retrieval Recall@k and MRR@k
docs/
  ai-engineering-guide.md  Guided tour of the architecture and experiments
tests/            Unit tests for ingestion, retrieval, metrics, and grounding
```

## Learn the AI engineering patterns

This repository is designed as a working reference, not only a finance Q&A
demo. Follow the [AI engineering guide](docs/ai-engineering-guide.md) to trace
a request through ingestion, retrieval, agent orchestration, grounding, and
evaluation, with suggested experiments at each stage.

## Web search fallback

When the agent exhausts local retrieval (no documents ingested for that company/period), it can fall back to a live web search, disabled by default.

```bash
# .env
WEB_SEARCH_ENABLED=true

# Tavily (recommended — finance-aware result ranking, free tier available)
TAVILY_API_KEY=tvly-...

# DuckDuckGo is used automatically if TAVILY_API_KEY is not set (no key needed)
```

The agent only triggers web search after confirming via `list_documents` that local data is absent. Web-sourced claims are always cited as `[WEB: <url>]` and flagged as less authoritative than ingested filings.

## Citation format

Every factual claim appears in the answer's `claims` array with its classification
and citation labels:

- `[AAPL · FY2024 Q3 · p.4]` — text retrieved from a document chunk
- `[STRUCTURED DATA: AAPL metrics table]` — value from the SQL metrics table
- `[CALCULATED — calculate(yoy_growth)]` — derived by the Python tool

Forward-looking statements are flagged separately so reported facts and management guidance are never mixed.

If grounding review fails, the API returns `grounding_status: "unverified"`, a
safe JSON-formatted `answer`, the rejected JSON answer in `unverified_draft`, and
specific problems in `grounding_issues`. Do not rely on the rejected draft as factual.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Evaluating narrative retrieval

After applying migrations and ingesting the demo reports, run the labeled
retrieval benchmark:

```bash
python -m scripts.eval_rag --k 8
```

It reports macro Recall@k and MRR@k for evidence-labeled guidance and narrative
queries. The benchmark requires the configured database and embedding provider.
New section context is included in chunks created after this change; re-ingest
existing PDFs to apply it to already stored chunks.
