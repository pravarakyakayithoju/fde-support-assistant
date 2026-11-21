import html
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import agent
from .db import conn
from .ingest import ingest

app = FastAPI(title="FDE Support Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    question: str


class IngestIn(BaseModel):
    sources: list[str]  # URLs or server-side file paths


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(body: ChatIn):
    return agent.run(body.question)


@app.post("/ingest")
def ingest_sources(body: IngestIn):
    return {s: ingest(s) for s in body.sources}


@app.get("/traces")
def traces(limit: int = 50):
    with conn() as c:
        return c.execute("SELECT * FROM traces ORDER BY id DESC LIMIT %s", (limit,)).fetchall()


@app.get("/admin", response_class=HTMLResponse)
def admin():
    """Bare-bones trace viewer. Replace with a React page later if you want."""
    with conn() as c:
        rows = c.execute("SELECT * FROM traces ORDER BY id DESC LIMIT 100").fetchall()
        stats = c.execute(
            """SELECT count(*) n, round(avg(latency_ms)) avg_ms,
                      percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) p95_ms,
                      coalesce(sum(cost_usd),0) cost, count(error) errors
               FROM traces WHERE created_at > now() - interval '24 hours'"""
        ).fetchone()
    e = lambda x: html.escape(str(x if x is not None else ""))
    trs = "".join(
        f"<tr><td>{r['id']}</td><td>{e(r['created_at'])[:19]}</td><td>{e(r['question'])}</td>"
        f"<td>{e(r['answer'])[:300]}</td>"
        f"<td>{'<br>'.join(e(t['name']) + ' ' + e(t['ms']) + 'ms' for t in (r['tool_calls'] or []))}</td>"
        f"<td>{'<br>'.join(e(x['source']).split('/')[-1] + ' (' + e(x['score']) + ')' for x in (r['retrieved'] or []))}</td>"
        f"<td>{r['latency_ms']}</td><td>{r['input_tokens']}/{r['output_tokens']}</td>"
        f"<td>${float(r['cost_usd'] or 0):.5f}</td><td style='color:#b00'>{e(r['error'])}</td></tr>"
        for r in rows
    )
    return f"""<!doctype html><html><head><title>Traces</title><style>
      body{{font:13px system-ui;margin:24px}} table{{border-collapse:collapse;width:100%}}
      td,th{{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}}
      th{{background:#f5f5f5}} .k{{display:inline-block;margin-right:24px}}</style></head><body>
      <h2>Request traces</h2>
      <p><span class=k>Last 24h: <b>{stats['n']}</b> requests</span>
         <span class=k>avg <b>{stats['avg_ms'] or 0}</b> ms</span>
         <span class=k>p95 <b>{round(stats['p95_ms'] or 0)}</b> ms</span>
         <span class=k>cost <b>${float(stats['cost']):.4f}</b></span>
         <span class=k>errors <b>{stats['errors']}</b></span></p>
      <table><tr><th>#</th><th>time</th><th>question</th><th>answer</th><th>tools</th>
      <th>retrieved (score)</th><th>ms</th><th>tok in/out</th><th>cost</th><th>error</th></tr>
      {trs}</table></body></html>"""
