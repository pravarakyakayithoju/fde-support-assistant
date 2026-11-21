from .config import TOP_K
from .db import conn
from .llm import embed


RRF_K = 60       # standard reciprocal-rank-fusion constant
CANDIDATES = 20  # how many results each search contributes before fusion


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search: vector similarity + Postgres full-text, merged with reciprocal rank fusion.
    Vector search catches paraphrases; keyword search catches exact terms ("application fee", "TC")."""
    qv = embed([query], task="RETRIEVAL_QUERY")[0]
    with conn() as c:
        return c.execute(
            """
            WITH vec AS (
                SELECT id, row_number() OVER (ORDER BY embedding <=> %(qv)s) AS r,
                       1 - (embedding <=> %(qv)s) AS sim
                FROM chunks ORDER BY embedding <=> %(qv)s LIMIT %(n)s
            ),
            q AS (  -- OR the words together so partial matches still count
                SELECT to_tsquery('english',
                       replace(plainto_tsquery('english', %(q)s)::text, '&', '|')) AS tq
            ),
            kw AS (
                SELECT ch.id, row_number() OVER (ORDER BY ts_rank_cd(ch.tsv, q.tq) DESC) AS r
                FROM chunks ch, q WHERE ch.tsv @@ q.tq
                ORDER BY ts_rank_cd(ch.tsv, q.tq) DESC LIMIT %(n)s
            ),
            fused AS (
                SELECT coalesce(vec.id, kw.id) AS id,
                       coalesce(1.0 / (%(rrf)s + vec.r), 0) + coalesce(1.0 / (%(rrf)s + kw.r), 0) AS rrf,
                       vec.sim, vec.r AS vec_rank, kw.r AS kw_rank
                FROM vec FULL OUTER JOIN kw ON vec.id = kw.id
            )
            SELECT ch.id AS chunk_id, ch.content, ch.page, d.source, d.title,
                   f.rrf AS score, f.sim AS vector_sim, f.vec_rank, f.kw_rank
            FROM fused f JOIN chunks ch ON ch.id = f.id JOIN documents d ON d.id = ch.document_id
            ORDER BY f.rrf DESC
            LIMIT %(k)s
            """,
            {"qv": qv, "q": query, "n": CANDIDATES, "rrf": RRF_K, "k": k},
        ).fetchall()


def format_context(chunks: list[dict]) -> str:
    """Numbered passages so the model can cite [1], [2]..."""
    parts = []
    for i, ch in enumerate(chunks, 1):
        loc = f"{ch['title']}" + (f", p.{ch['page']}" if ch["page"] else "")
        parts.append(f"[{i}] ({loc})\n{ch['content']}")
    return "\n\n---\n\n".join(parts)
