import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/assistant")

# Which service answers questions: "groq" or "gemini"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
# Which service makes embeddings: "local" (free, runs on your laptop) or "gemini"
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local").lower()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

_default_chat = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-3.8-flash"}
CHAT_MODEL = os.environ.get("CHAT_MODEL", _default_chat.get(LLM_PROVIDER, ""))
# Model used to grade evals. A smaller model has its own separate rate limit on Groq.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", CHAT_MODEL)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")          # used when EMBED_PROVIDER=gemini
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")  # 768 dims, used when local
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))  # must match schema.sql

BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "Example College Admissions Office")

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150
TOP_K = 5

# USD per 1M tokens — update from your provider's pricing page for your model.
PRICE_IN_PER_M = float(os.environ.get("PRICE_IN_PER_M", "0.15"))
PRICE_OUT_PER_M = float(os.environ.get("PRICE_OUT_PER_M", "0.60"))
