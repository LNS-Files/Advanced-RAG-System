from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
import ollama
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)
DEFAULT_DB_DIR = Path(__file__).resolve().parents[1] / "chroma_db"
DEFAULT_COLLECTION_NAME = "document_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "Answer the question based ONLY on the provided context. "
    "If the answer cannot be found, state that you do not know."
)


def retrieve_context(query: str, top_k: int = 3) -> str:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    try:
        embedding_client = _initialize_embedding_client()
        collection = _load_collection()
        query_embedding = _embed_query(embedding_client, query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("Failed to retrieve context for query: %s", exc)
        return ""
    except Exception as exc:
        LOGGER.exception("Unexpected retrieval failure: %s", exc)
        return ""

    documents = _extract_documents(results)
    return "\n\n".join(documents)


def generate_answer(user_query: str, context: str) -> str:
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    try:
        response = ollama.chat(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        f"{context.strip() or 'No context was retrieved.'}\n\n"
                        "Question:\n"
                        f"{user_query.strip()}"
                    ),
                },
            ],
        )
    except (ollama.ResponseError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Failed to generate answer: %s", exc)
        return "I do not know."
    except Exception as exc:
        LOGGER.exception("Unexpected answer generation failure: %s", exc)
        return "I do not know."

    answer = _extract_ollama_message(response)
    return answer.strip() if answer else "I do not know."


def _initialize_embedding_client() -> SentenceTransformer:
    try:
        return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError("Embedding model could not be initialized") from exc


def _load_collection(
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' could not be loaded"
        ) from exc


def _embed_query(client: SentenceTransformer, query: str) -> list[float]:
    embedding = client.encode(
        query.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist()


def _extract_documents(results: dict[str, Any]) -> list[str]:
    raw_documents = results.get("documents") or []
    if not raw_documents:
        return []

    first_result_set = raw_documents[0] if isinstance(raw_documents[0], list) else raw_documents
    return [
        document.strip()
        for document in first_result_set
        if isinstance(document, str) and document.strip()
    ]


def _extract_ollama_message(response: Any) -> str:
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return str(response.message.content)

    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))

    return ""


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
