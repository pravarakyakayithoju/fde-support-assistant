# FDE Support Assistant

A RAG + tool-using agent that answers questions from one organisation's documents **with citations**, looks up records, raises support tickets and emails summaries. Every request is traced (retrieved chunks, tool calls, latency, tokens, cost) and quality is measured with an eval suite.

The included data is a **fictional** college admissions office ("Northfield Institute of Technology"): 6 PDFs covering admission rules, fees, documents, scholarships, hostel rules and FAQs.

**Stack:** Python, FastAPI, Postgres + pgvector, Groq (`gpt-oss-120b`) or Gemini for the LLM, local `bge-base-en-v1.5` embeddings (fastembed), Docker.

## Results

30-question eval set (22 answerable, 4 that must be declined, 4 tool-use), graded by an LLM judge:

| Metric | Score |
|---|---|
| Answer accuracy | 80% |
| Hallucination rate | 0% |
| Citation rate | 100% |
| Tool-selection accuracy | 100% |
| Correctly declined unanswerable questions | 4 / 4 |

### What the evals caught

| Change | Accuracy (first 5 q) | Citations |
|---|---|---|
| Baseline (paragraph chunking, vector search) | 60% | 0% |
| Sentence-aware chunking + citation format fix | 40% | 60% |
| Contextual chunks + hybrid search | **100%** | **100%** |

Retrieval traces showed that the first chunk of every PDF (document header + boilerplate) ranked top for almost every question, while chunks holding the actual facts were never retrieved. Fixes:
- **Contextual chunks:** every chunk is prefixed with its document title, and repeated boilerplate is stripped.
- **Hybrid search:** pgvector similarity + Postgres full-text search, merged with reciprocal rank fusion (RRF).
- The 0% citation rate was a format mismatch (the model cited as `【1】`); answers are now normalised to `[1]`.

The remaining failures were answers that omitted a condition (e.g. "single rooms only from second year"), traced to a "be concise" instruction in the system prompt.

## How it works

```
question ─► agent (LLM with tools) ─► search_docs ─► hybrid retrieval (vector + keyword, RRF)
                 │                     lookup_record ─► Postgres
                 │                     create_ticket ─► Postgres
                 │                     email_summary ─► (simulated)
                 ▼
     answer with [n] citations  +  trace row (chunks, tool calls, latency, tokens, cost)
```

```
backend/
  schema.sql          documents, chunks (vector + tsvector), records, tickets, traces
  app/ingest.py       PDF/URL -> clean -> sentence-aware chunks with title -> embeddings
  app/rag.py          hybrid retrieval with reciprocal rank fusion
  app/tools.py        tool definitions (shared by Groq and Gemini) + implementations
  app/agent.py        tool-calling loop for Groq or Gemini, with tracing
  app/main.py         /chat  /ingest  /traces  /admin
  debug_retrieval.py  index vs exact-search ranking for one question
evals/
  questions.jsonl     30 test questions with expected answers
  run_evals.py        accuracy, hallucination, citation and tool metrics
```

## Run locally

Needs Docker, Python 3.11+, and a Groq API key (console.groq.com).

```bash
cp .env.example backend/.env       # Windows: copy .env.example backend\.env  — then add your key
docker compose up -d               # Postgres + pgvector
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.ingest ../data/raw   # ingest the sample PDFs
uvicorn app.main:app --reload      # API docs: http://localhost:8000/docs, traces: /admin

python ../evals/run_evals.py                    # full eval
python ../evals/run_evals.py --ids q06,q16      # re-run specific questions
```

Switch providers in `.env`: `LLM_PROVIDER=groq|gemini`, `EMBED_PROVIDER=local|gemini`. `JUDGE_MODEL` lets the eval grader use a different model.

## Roadmap

- [ ] Deploy on Google Cloud (Cloud Run + Cloud SQL)
- [ ] Small React chat UI
- [ ] Reduce latency (p50 ≈ 11 s on the full set, mostly multi-step tool questions)
- [ ] Real email sending; auth
