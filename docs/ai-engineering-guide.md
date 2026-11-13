# AI Engineering Guide

This project is a small, runnable reference for studying how AI application
components fit together. It is not a general RAG framework: the code makes
specific tradeoffs for financial filings, and the guide points to those
tradeoffs so you can inspect, test, and change them.

## Learning path

| Topic | Start here | What to notice |
|---|---|---|
| API contracts | [Research schemas](../app/api/schemas.py), [research route](../app/api/routes/research.py) | Request validation, dependency-injected database sessions, response shaping |
| Agent orchestration | [Graph](../app/agent/graph.py) | Classification, routing, tool use, deterministic branches, and state passed between nodes |
| Document ingestion | [Extractor](../app/ingestion/extractor.py), [chunker](../app/ingestion/chunker.py), [pipeline](../app/ingestion/pipeline.py) | Extraction errors propagate into chunk quality; chunk text and metadata both affect retrieval |
| Retrieval | [Vector store](../app/retrieval/vector_store.py), [database models](../app/database/models.py) | Metadata filters, vector and full-text candidate generation, rank fusion, provenance |
| Grounding | [Prompts](../app/prompts/system.py), [grounding tests](../tests/test_grounding.py) | A grounded draft is checked against supplied evidence and revised once; an LLM evaluator is a safeguard, not a proof |
| Evaluation | [Retrieval benchmark](../scripts/eval_rag.py), [API eval harness](../scripts/eval.py), [retrieval metrics](../app/retrieval/evaluation.py) | Retrieval ranking and end-to-end answer behavior are separate evaluation problems |

## Trace one request

Start at `POST /research/query` and follow this sequence:

1. The route validates the request and calls `run_research_query`.
2. The graph classifies the question. A supported operating-margin comparison
   can take a deterministic SQL path; other questions enter the agent path.
3. Narrative retrieval embeds the question, searches with pgvector and
   PostgreSQL full-text search, then combines ranked candidates with reciprocal
   rank fusion.
4. The agent receives retrieved evidence and can call tools for structured
   metrics, calculations, discovery, or (when enabled) web search.
5. Synthesis produces an answer from the collected evidence. The grounding
   evaluator checks factual claims and permits one revision attempt.

Trace the matching source files above rather than treating the prompt as the
whole system. Retrieval bugs, extraction errors, and routing mistakes can all
look like generation errors from the outside.

## Why use both RAG and SQL?

Narrative retrieval is useful for questions about explanations, guidance, and
management commentary. It is a poor way to reconstruct complete numeric time
series: vector similarity can omit a quarter or mix periods. This project
stores metrics as relational records and uses SQL for supported numerical
comparisons, keeping calculations deterministic where possible.

For narrative text, hybrid retrieval addresses two complementary signals:

- Vector similarity helps find passages that express an idea in different
  words.
- PostgreSQL full-text search helps find exact names and terms.
- Reciprocal rank fusion combines their rankings without assuming that their
  raw scores have the same scale.

The current fusion is deliberately simple. It is not a cross-encoder reranker,
and its ranking quality should be judged with the benchmark rather than
assumed.

## Run and inspect the system

Follow the [README quick start](../README.md#quick-start) to start the API and
seed the synthetic reports. Then:

```bash
# Watch a request move through the application
docker compose logs -f --tail=100 api

# Run the retrieval benchmark against the configured database and embedder
python -m scripts.eval_rag --k 8
```

Inspect API contracts and responses at <http://localhost:8000/docs>.

The benchmark reports Recall@k and MRR@k for a small set of evidence-labeled
queries. Its labels are based on expected phrases in the synthetic demo corpus;
they are a teaching fixture, not a representative or human-annotated finance
benchmark. Expand the cases and inspect their gold-evidence criteria before
using the scores to make broader quality claims.

## Experiments

Make one change at a time, keep the corpus and embedding model fixed, and record
both the metric and the qualitative failures you observe.

1. **Top-k tradeoff:** compare `python -m scripts.eval_rag --k 4`,
   `python -m scripts.eval_rag --k 8`, and `python -m scripts.eval_rag --k 12`.
   Compare Recall@k and MRR@k. More context can improve recall but also adds
   irrelevant evidence and synthesis cost.
2. **Chunking:** vary `CHUNK_SIZE` and `CHUNK_OVERLAP` in configuration, re-ingest
   the same PDFs, and rerun the benchmark. Existing chunks are not rewritten
   automatically when chunking code or settings change.
3. **Metadata filters:** compare an unfiltered query with ticker, fiscal-period,
   or section filters. Check whether precision improves without excluding the
   evidence needed to answer the question.
4. **Evaluation coverage:** add a benchmark case for a difficult query and
   identify its relevant evidence phrases before changing retrieval. Include
   a missing-evidence case to verify that the system abstains rather than
   inventing support.
5. **Grounding behavior:** change one synthesis instruction and run the
   grounding tests. Look at both the accepted answer and the rejected draft
   path; passing a prompt test does not establish general model reliability.

The chunk size used by the text splitter is character-based. Stored
`token_count` values use `cl100k_base`; for Ollama models this is an estimate,
not the model's exact tokenizer count.

## Questions to ask while reading

- At which stage could a relevant fact be lost: PDF extraction, chunking,
  metadata filtering, ranking, prompt assembly, or synthesis?
- Does a citation identify the exact source passage or only a document?
- Which claims should use structured data rather than narrative retrieval?
- How would you calibrate the similarity threshold for a new embedding model?
- What would a larger, human-labeled benchmark need to measure beyond Recall@k
  and MRR@k?
- How would you compare reciprocal rank fusion with a learned reranker without
  changing several variables at once?

