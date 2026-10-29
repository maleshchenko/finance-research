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

> **Note on embedding dimensions:** the `VECTOR_DIMENSION` setting must match
> the embedding model and must be set **before** the first `alembic upgrade head`
> — pgvector columns are fixed-width. If you need to change it later, drop and
> recreate the database.

Interactive API docs: **http://localhost:8000/docs**

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
