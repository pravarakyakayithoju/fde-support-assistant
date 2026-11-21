# FDE Support Assistant

A RAG + tool-using agent that answers questions from one organisation's documents **with citations**, looks up records, raises support tickets and emails summaries. Every request is traced (retrieved chunks, tool calls, latency, tokens, cost) and quality is measured with an eval suite.

The included data is a **fictional** college admissions office ("Northfield Institute of Technology"): 6 PDFs covering admission rules, fees, documents, scholarships, hostel rules and FAQs.

**Stack:** Python, FastAPI, Postgres + pgvector, Groq (`gpt-oss-120b`) or Gemini for the LLM, local `bge-base-en-v1.5` embeddings (fastembed), Docker.

## Results

30-question eval set (22 answerable, 4 that must be declined, 4 tool-use), graded by an LLM judge:

| Metric | Score |
|---|---|
| Answer accuracy | 86.7% (26/30) |
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

On the full set, accuracy went from 80% to 86.7% after replacing a "be concise" instruction with "complete but brief: include every amount, condition or exception that affects the answer".

### Known failure modes (latest run)
Retrieval found the right passage in every remaining failure; the misses are in answer generation:
- **Omitted conditions** (q06, q16): the answer gives the main fact but drops a qualifying rule present in a retrieved passage, e.g. "single rooms only from second year".
- **Under-reporting tool results** (q30): the email was sent with the right content, but the reply didn't say what it contained.
- **Strict expectation** (q28): the answer correctly says the fee is paid but doesn't add the admission status the test expects.

These were not tuned further on purpose: with 30 questions, more prompt tweaks risk overfitting the eval set. Next step is a larger eval set before changing the prompt again.

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
- [ ] Reduce latency (p50 ≈ 12.7 s on the full set, mostly multi-step tool questions)
- [ ] Grow the eval set and add a completeness check for answers
- [ ] Real email sending; auth
