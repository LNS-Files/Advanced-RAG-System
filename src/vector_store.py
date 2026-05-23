from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import BATCH_SIZE, COLLECTION_NAME, DB_DIR

LOGGER = logging.getLogger(__name__)


def initialize_vector_db(
    persist_directory: str | Path = DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    db_path = Path(persist_directory)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks_to_db(
    chunks: Sequence[str],
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    *,
    source: str | None = None,
    per_chunk_metadata: list[dict[str, Any]] | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    clean_chunks = [c.strip() for c in chunks if c and c.strip()]
    if not clean_chunks:
        return []

    stored_ids: list[str] = []

    for batch_start, batch in enumerate(_batched(clean_chunks, batch_size)):
        offset_base = batch_start * batch_size
        embeddings = _generate_embeddings(embedding_model, batch)
        ids = [
            _stable_chunk_id(chunk=chunk, source=source, index=offset_base + i)
            for i, chunk in enumerate(batch)
        ]
        metadatas: list[dict[str, Any]] = []
        for i in range(len(batch)):
            global_i = offset_base + i
            meta: dict[str, Any] = {"chunk_index": global_i}
            if source:
                meta["source"] = source
            if per_chunk_metadata and global_i < len(per_chunk_metadata):
                meta.update(per_chunk_metadata[global_i])
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=list(batch),
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stored_ids.extend(ids)
        LOGGER.info("Stored %d chunks in vector database", len(batch))

    return stored_ids


def _generate_embeddings(model: SentenceTransformer, texts: Sequence[str]) -> list[list[float]]:
    return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()


def _stable_chunk_id(chunk: str, source: str | None, index: int) -> str:
    prefix = source or "unknown_source"
    return hashlib.sha256(f"{prefix}:{index}:{chunk}".encode()).hexdigest()


def _batched(items: Sequence[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def fetch_all_documents(collection: chromadb.Collection) -> list[dict[str, Any]]:
    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    records: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        if not (isinstance(doc, str) and doc.strip()):
            continue
        meta: dict[str, Any] = metas[i] if metas and i < len(metas) else {}
        records.append({
            "text": doc.strip(),
            "source": str(meta.get("source", "Unknown")),
            "page": meta.get("page"),
        })
    return records
