"""Compare index search vs exact search for a question, showing every chunk's score.

Usage:  python debug_retrieval.py "What is the last date to submit the online application?"
"""
import sys

from app.db import conn, pool
from app.llm import embed

q = sys.argv[1] if len(sys.argv) > 1 else "What is the last date to submit the online application?"
qv = embed([q], task="RETRIEVAL_QUERY")[0]
SQL = """SELECT d.title, ch.chunk_index, round((1 - (ch.embedding <=> %s))::numeric, 3) AS score,
                left(ch.content, 60) AS start
         FROM chunks ch JOIN documents d ON d.id = ch.document_id
         ORDER BY ch.embedding <=> %s LIMIT %s"""

try:
    with conn() as c:
        print(f"\nQUESTION: {q}\n\n--- using the HNSW index (what the app does), top 5 ---")
        for r in c.execute(SQL, (qv, qv, 5)).fetchall():
            print(f"{r['score']}  {r['title']}#{r['chunk_index']}  {r['start']}")
        c.execute("SET enable_indexscan = off")
        print("\n--- exact search over ALL chunks ---")
        for r in c.execute(SQL, (qv, qv, 100)).fetchall():
            print(f"{r['score']}  {r['title']}#{r['chunk_index']}  {r['start']}")
finally:
    pool.close()
