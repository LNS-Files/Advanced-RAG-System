from __future__ import annotations

import hashlib
import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    DB_DIR,
    DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    QUERY_REWRITE_PROMPT,
    RRF_K,
    SYSTEM_PROMPT,
    TOP_K,
)
from src.vector_store import fetch_all_documents

LOGGER = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None


@dataclass
class _BM25Cache:
    bm25: BM25Okapi
    records: list[dict[str, Any]]
    count: int


_bm25_cache: _BM25Cache | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            raise RuntimeError("Embedding model could not be initialized") from exc
    return _embedding_model


def _load_collection(
    persist_directory: str | Path = DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' could not be loaded"
        ) from exc


def _embed_query(model: SentenceTransformer, query: str) -> list[float]:
    return model.encode(
        query.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def _extract_results(
    results: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    raw_docs = results.get("documents") or []
    raw_dists = results.get("distances") or []
    raw_metas = results.get("metadatas") or []
    if not raw_docs:
        return []

    docs: list[Any] = raw_docs[0] if isinstance(raw_docs[0], list) else raw_docs
    dists: list[Any] = raw_dists[0] if raw_dists and isinstance(raw_dists[0], list) else raw_dists
    metas: list[Any] = raw_metas[0] if raw_metas and isinstance(raw_metas[0], list) else raw_metas

    filtered: list[dict[str, Any]] = []
    for i, (doc, dist) in enumerate(zip(docs, dists or [None] * len(docs))):
        if not (isinstance(doc, str) and doc.strip()):
            continue
        if dist is not None and dist > threshold:
            continue
        meta: dict[str, Any] = metas[i] if metas and i < len(metas) else {}
        filtered.append({
            "text": doc.strip(),
            "source": str(meta.get("source", "Unknown")),
            "page": meta.get("page"),
        })
    return filtered


def rewrite_query(query: str, history: list[dict] | None = None) -> str:
    if not query.strip():
        return query

    history_text = ""
    if history:
        for msg in history[-6:]:  # last 3 user+assistant turns
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    user_content = (
        f"Conversation history:\n{history_text}\n" if history_text else ""
    ) + f"Original question: {query.strip()}\nRewritten question:"

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        rewritten = _extract_ollama_message(response).strip()
        return rewritten if rewritten else query
    except Exception as exc:
        LOGGER.warning("Query rewriting failed, using original query: %s", exc)
        return query


def _get_bm25_index(collection: chromadb.Collection) -> _BM25Cache:
    global _bm25_cache
    count = collection.count()
    if _bm25_cache is None or _bm25_cache.count != count:
        records = fetch_all_documents(collection)
        tokenized = [r["text"].lower().split() for r in records] or [[""]]
        _bm25_cache = _BM25Cache(
            bm25=BM25Okapi(tokenized),
            records=records,
            count=count,
        )
    return _bm25_cache


def _bm25_search(
    query: str,
    cache: _BM25Cache,
    top_n: int,
) -> list[dict[str, Any]]:
    if not cache.records:
        return []
    scores = cache.bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return [cache.records[i] for i in ranked[:top_n] if scores[i] > 0]


def _reciprocal_rank_fusion(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    top_k: int,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    id_to_record: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vector_hits):
        doc_id = hashlib.md5(hit["text"].encode()).hexdigest()
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_record[doc_id] = hit
    for rank, hit in enumerate(bm25_hits):
        doc_id = hashlib.md5(hit["text"].encode()).hexdigest()
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_record[doc_id] = hit
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [id_to_record[doc_id] for doc_id in sorted_ids[:top_k]]


def retrieve_context(
    query: str,
    top_k: int = TOP_K,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> tuple[str, list[dict[str, Any]]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    try:
        model = _get_embedding_model()
        collection = _load_collection()

        total_docs = collection.count()
        if total_docs == 0:
            return "", []

        # Vector search — fetch 2× candidates so RRF has enough material
        candidate_k = min(top_k * 2, total_docs)
        query_embedding = _embed_query(model, query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        vector_hits = _extract_results(results, distance_threshold)

        # BM25 keyword search over full corpus
        bm25_cache = _get_bm25_index(collection)
        bm25_hits = _bm25_search(query, bm25_cache, top_n=top_k * 2)

        # Fuse both ranked lists with RRF
        hits = _reciprocal_rank_fusion(vector_hits, bm25_hits, top_k=top_k)

    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("Failed to retrieve context: %s", exc)
        return "", []
    except Exception as exc:
        LOGGER.exception("Unexpected retrieval failure: %s", exc)
        return "", []

    if not hits:
        LOGGER.info("No documents found for query")

    context = "\n\n".join(h["text"] for h in hits)

    seen: set[tuple[str, Any]] = set()
    citations: list[dict[str, Any]] = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            citations.append({"source": h["source"], "page": h["page"]})

    return context, citations


def generate_answer(
    user_query: str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    messages = _build_messages(user_query, context, history)

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=CHAT_MODEL,
            messages=messages,
        )
    except (ollama.ResponseError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Failed to generate answer: %s", exc)
        return "I do not know."
    except Exception as exc:
        LOGGER.exception("Unexpected answer generation failure: %s", exc)
        return "I do not know."

    return _extract_ollama_message(response).strip() or "I do not know."


def _extract_ollama_message(response: Any) -> str:
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return str(response.message.content)
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""


def _build_messages(
    user_query: str,
    context: str,
    history: list[dict] | None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            "Context:\n"
            f"{context.strip() or 'No context was retrieved.'}\n\n"
            "Question:\n"
            f"{user_query.strip()}"
        ),
    })
    return messages


def stream_answer(
    user_query: str,
    context: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    if not user_query.strip():
        yield "I do not know."
        return

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        messages = _build_messages(user_query, context, history)
        for chunk in client.chat(model=CHAT_MODEL, messages=messages, stream=True):
            content = _extract_ollama_message(chunk)
            if content:
                yield content
    except (ollama.ResponseError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Streaming answer failed: %s", exc)
        yield "I do not know."
    except Exception as exc:
        LOGGER.exception("Unexpected streaming failure: %s", exc)
        yield "I do not know."


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
