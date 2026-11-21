"""Model clients. Each one is created on first use, so you only need keys for what you use."""
import numpy as np

from .config import (EMBED_DIM, EMBED_MODEL, EMBED_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY,
                     LLM_PROVIDER, CHAT_MODEL, JUDGE_MODEL, LOCAL_EMBED_MODEL)

_clients = {}


def gemini():
    if "gemini" not in _clients:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        from google import genai
        _clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
    return _clients["gemini"]


def groq():
    if "groq" not in _clients:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        from openai import OpenAI  # Groq speaks the OpenAI API format
        _clients["groq"] = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _clients["groq"]


def _local_embedder():
    if "local" not in _clients:
        from fastembed import TextEmbedding  # downloads the model (~200 MB) on first use
        _clients["local"] = TextEmbedding(model_name=LOCAL_EMBED_MODEL)
    return _clients["local"]


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[np.ndarray]:
    """task: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    if EMBED_PROVIDER == "local":
        m = _local_embedder()
        gen = m.query_embed(texts) if task == "RETRIEVAL_QUERY" else m.passage_embed(texts)
        vecs = [np.array(v, dtype=np.float32) for v in gen]
    else:
        from google.genai import types
        vecs = []
        for i in range(0, len(texts), 100):  # API batch limit
            resp = gemini().models.embed_content(
                model=EMBED_MODEL, contents=texts[i:i + 100],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
            )
            vecs += [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
    for v in vecs:
        if len(v) != EMBED_DIM:
            raise ValueError(f"Embedding has {len(v)} dims but EMBED_DIM/schema is {EMBED_DIM}")
    return [v / np.linalg.norm(v) for v in vecs]


def complete_json(prompt: str) -> str:
    """One-shot JSON completion, used by the eval judge."""
    if LLM_PROVIDER == "groq":
        r = groq().chat.completions.create(
            model=JUDGE_MODEL, temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return r.choices[0].message.content or ""
    from google.genai import types
    r = gemini().models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    return r.text or ""
