# Financial Research Assistant

A research assistant that answers questions about financial reports, earnings releases, and economic data, citing every claim.

Built with **LangChain**, **LangGraph**, **PostgreSQL + pgvector**, and **FastAPI**.

## Why not pure RAG?

Semantic search retrieves relevant paragraphs but struggles to reconstruct a complete numerical series. This project uses a hybrid approach:

| Question type | Strategy |
|---|---|
| "What did management say about margins?" | RAG — pgvector cosine search |
| "How did revenue change over 8 quarters?" | SQL query -> Python calculation |
| "Compare margins and explain the trend" | Both, then LLM synthesis |

The LLM never guesses numerical values. It retrieves or calculates them, then synthesises a cited answer.

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
  -d '{"question": "Compare operating margins for AAPL, MSFT, and GOOGL across FY2023"}' \
  | jq .answer
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

### Example console output with local Ollama:

```
curl -X POST http://localhost:8000/research/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Compare operating margins for AAPL, MSFT, and GOOGL across all four quarters of FY2023"}'

{"answer":"**Operating Margin Comparison Across Companies and Periods**\n\n### **Companies and Periods Compared**\n\n| Company | Q4 FY2023 | Q1 FY2023 | Q3 FY2023 | Q2 FY2023 |\n| --- | --- | --- | --- | --- |\n| AAPL | 33.4% | 32.0% | 33.4% | 24.6% |\n| MSFT |  |  |  |  |\n| GOOGL | 27.7% | 30.6% | 29.3% | 24.6% |\n\n### **Data Gaps and Limitations**\n\n* MSFT's Q1 FY2023 operating margin is not reported in the provided evidence.\n* Web search was attempted to retrieve MSFT's Q1 FY2023 operating margin, but no reliable data was found.\n\n### **Calculated Results**\n\n| Company | Q4 FY2023 | Q1 FY2023 | Q3 FY2023 | Q2 FY2023 |\n| --- | --- | --- | --- | --- |\n| AAPL | 33.4% | 32.0% | 33.4% | 24.6% |\n| MSFT |  |  |  |  |\n| GOOGL | 27.7% | 30.6% | 29.3% | 24.6% |\n\n### **Citations**\n\n* AAPL · FY2023 Q4 · p.1: \"Operating income of 39.1 billion reflects an operating margin of 33.4%, compared to 32.2% in the prior-year period.\"\n* AAPL · FY2023 Q1 · p.1: \"Operating income of 30.3 billion reflects an operating margin of 32.0%, compared to 30.8% in the prior-year period.\"\n* GOOGL · FY2023 Q4 · p.1: \"Operating income of 23.9 billion reflects an operating margin of 27.7%, compared to 26.5% in the prior-year period.\"\n* GOOGL · FY2023 Q1 · p.1: \"Operating income of 21.3 billion reflects an operating margin of 30.6%, compared to 29.4% in the prior-year period.\"\n* GOOGL · FY2023 Q3 · p.1: \"Operating income of 21.8 billion reflects an operating margin of 29.3%, compared to 28.1% in the prior-year period.\"\n* GOOGL · FY2023 Q2 · p.1: \"Operating income of 17.1 billion reflects an operating margin of 24.6%, compared to 23.4% in the prior-year period.\"\n\n### **Interpretation**\n\nThe operating margins for AAPL, MSFT, and GOOGL vary across different periods. AAPL's operating margin is consistently higher than GOOGL's, while MSFT's operating margin is not reported for Q1 FY2023.\n\n### **Web Source**\n\nMSFT's Q1 FY2023 operating margin was not found in the provided evidence or through a web search.","classification":"HYBRID","citations":[],"tool_calls_made":["search_documents","query_metrics"],"elapsed_seconds":207.28,"error":null}
```

> **Note on embedding dimensions:** the `VECTOR_DIMENSION` setting must match
> the embedding model and must be set **before** the first `alembic upgrade head`
> — pgvector columns are fixed-width. If you need to change it later, drop and
> recreate the database.

Interactive API docs: **http://localhost:8000/docs**

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
  retrieval/      pgvector semantic search + typed SQL metric queries
  agent/          LangGraph graph, 5 LangChain tools
  api/routes/     POST /ingest/upload, POST /research/query, GET /documents/…
  database/       SQLAlchemy models, Alembic migrations
scripts/
  seed_demo.py    Generate and ingest synthetic reports
tests/            30 unit tests (no DB or API key required)
```

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

Every factual claim in an answer is tagged:

- `[AAPL · FY2024 Q3 · p.4]` — text retrieved from a document chunk
- `[STRUCTURED DATA: AAPL metrics table]` — value from the SQL metrics table
- `[CALCULATED — calculate(yoy_growth)]` — derived by the Python tool

Forward-looking statements are flagged separately so reported facts and management guidance are never mixed.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
