from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)
DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
DEFAULT_COLLECTION_NAME = "document_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 100


def initialize_vector_db(
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    db_path = Path(persist_directory)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks_to_db(
    chunks: Sequence[str],
    collection: Collection,
    *,
    source: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not clean_chunks:
        return []

    embedding_client = SentenceTransformer(embedding_model)
    stored_ids: list[str] = []

    for batch_start, batch in enumerate(_batched(clean_chunks, batch_size)):
        embeddings = _generate_embeddings(
            client=embedding_client,
            texts=batch,
        )
        ids = [
            _stable_chunk_id(chunk=chunk, source=source, index=batch_start * batch_size + offset)
            for offset, chunk in enumerate(batch)
        ]
        metadatas = [
            _build_metadata(source=source, chunk_index=batch_start * batch_size + offset)
            for offset in range(len(batch))
        ]

        collection.upsert(
            ids=ids,
            documents=list(batch),
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stored_ids.extend(ids)
        LOGGER.info("Stored %s chunks in vector database", len(batch))

    return stored_ids


def _generate_embeddings(
    client: SentenceTransformer,
    texts: Sequence[str],
) -> list[list[float]]:
    embeddings = client.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _stable_chunk_id(chunk: str, source: str | None, index: int) -> str:
    source_prefix = source or "unknown_source"
    digest = hashlib.sha256(f"{source_prefix}:{index}:{chunk}".encode("utf-8")).hexdigest()
    return digest


def _build_metadata(source: str | None, chunk_index: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {"chunk_index": chunk_index}
    if source:
        metadata["source"] = source
    return metadata


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
