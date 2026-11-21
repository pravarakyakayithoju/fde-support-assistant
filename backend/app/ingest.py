"""Ingest PDFs and web pages -> chunks -> embeddings -> pgvector.

Usage:
    python -m app.ingest ../data/raw https://example.edu/admissions/faq
"""
import glob
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .db import conn, ensure_schema, pool
from .llm import embed


def load_pdf(path: str) -> tuple[str, list[tuple[int, str]]]:
    reader = PdfReader(path)
    pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    return meta_title.strip() or Path(path).stem.replace("_", " "), pages


def load_url(url: str) -> tuple[str, list[tuple[int | None, str]]]:
    html = httpx.get(url, timeout=30, follow_redirects=True).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    return title, [(None, soup.get_text("\n"))]


# Lines repeated on every document add noise to every chunk. Add patterns for your customer's docs.
BOILERPLATE = [
    r"SAMPLE DOCUMENT[^\n]*",
    r"Northfield Institute of Technology\s*\n",
]


def clean(text: str) -> str:
    for pat in BOILERPLATE:
        text = re.sub(pat, " ", text)
    return text


def split(text: str) -> list[str]:
    """Sentence-aware splitter: packs whole sentences up to CHUNK_SIZE characters,
    and repeats the last sentence(s) (up to CHUNK_OVERLAP chars) at the start of the next chunk.
    PDF text often has no blank lines between paragraphs, so we never cut mid-sentence."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)          # single line breaks = wrapped lines
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sents = []
    for para in paras:
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", para):  # "Rs. 1,200" is not split
            while len(sent) > CHUNK_SIZE:                         # pathological long sentence
                sents.append(sent[:CHUNK_SIZE]); sent = sent[CHUNK_SIZE:]
            if sent.strip():
                sents.append(sent.strip())

    chunks, cur = [], []
    for sent in sents:
        if cur and len(" ".join(cur)) + 1 + len(sent) > CHUNK_SIZE:
            chunks.append(" ".join(cur))
            keep = []
            for t in reversed(cur):
                if len(" ".join([t] + keep)) > CHUNK_OVERLAP:
                    break
                keep.insert(0, t)
            cur = keep
        cur.append(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def ingest(source: str) -> int:
    title, pages = load_url(source) if source.startswith("http") else load_pdf(source)
    # Contextual chunks: every chunk starts with its document title, so a chunk from the
    # middle of a document still "knows" what it is about (helps both vector and keyword search).
    rows = [(page, f"{title}\n{c}") for page, text in pages for c in split(clean(text))]
    if not rows:
        print(f"  ! no text extracted from {source} (scanned PDF? needs OCR)")
        return 0
    vectors = embed([c for _, c in rows])
    with conn() as c:
        c.execute("DELETE FROM documents WHERE source = %s", (source,))  # re-ingest = replace
        doc_id = c.execute(
            "INSERT INTO documents (source, title) VALUES (%s, %s) RETURNING id", (source, title)
        ).fetchone()["id"]
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, content, page, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [(doc_id, i, ch, pg, v) for i, ((pg, ch), v) in enumerate(zip(rows, vectors))],
            )
    return len(rows)


def expand(args: list[str]) -> list[str]:
    """Accept folders and *.pdf patterns (Windows cmd doesn't expand globs)."""
    out = []
    for a in args:
        if a.startswith("http"):
            out.append(a)
        elif Path(a).is_dir():
            out += sorted(str(p) for p in Path(a).glob("*.pdf"))
        elif any(ch in a for ch in "*?"):
            out += sorted(glob.glob(a))
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    ensure_schema()
    for src in expand(sys.argv[1:]):
        print(f"{src}: {ingest(src)} chunks")
    # Deletes + re-inserts can leave the HNSW graph with unreachable rows; rebuild it.
    with conn() as c:
        c.execute("REINDEX INDEX chunks_embedding_idx")
        n = c.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    print(f"index rebuilt; {n} chunks in database")
    pool.close()
