from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "document_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.2"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ENCODING_NAME = "cl100k_base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
TOP_K = 5
DISTANCE_THRESHOLD = 0.7
RRF_K = 60

SYSTEM_PROMPT = (
    "Answer the question based ONLY on the provided context. "
    "If the answer cannot be found, state that you do not know."
)

QUERY_REWRITE_PROMPT = (
    "You are a search query optimizer. Rewrite the user's question into a single, "
    "self-contained search query that will retrieve the most relevant document chunks.\n"
    "Rules:\n"
    "- Resolve pronouns and vague references (it, that, this, the above) using the conversation history\n"
    "- Make implicit topics explicit\n"
    "- Keep it concise — one sentence\n"
    "- Return ONLY the rewritten query, no explanation or preamble"
)
