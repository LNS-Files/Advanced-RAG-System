from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import tiktoken
from pypdf import PdfReader
from pypdf.errors import PdfReadError


LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_ENCODING = "cl100k_base"


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        reader = PdfReader(path)
    except (PdfReadError, OSError, ValueError) as exc:
        LOGGER.warning("Skipping unreadable PDF %s: %s", path, exc)
        return ""

    page_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (PdfReadError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Skipping page %s in %s due to extraction error: %s",
                page_number,
                path,
                exc,
            )
            continue

        if text.strip():
            page_text.append(text)

    return "\n\n".join(page_text)


def semantic_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    stride = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + chunk_size]
        if len(chunk_tokens) < chunk_size:
            break
        chunks.append(encoding.decode(chunk_tokens))

    return chunks


def iter_pdf_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> Iterable[Path]:
    path = Path(data_dir)
    if not path.exists():
        LOGGER.warning("Data directory does not exist: %s", path)
        return ()

    return sorted(path.glob("*.pdf"))


def run_ingestion_pipeline(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[Path, list[str]]:
    ingested_documents: dict[Path, list[str]] = {}

    for pdf_path in iter_pdf_files(data_dir):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            ingested_documents[pdf_path] = []
            continue

        ingested_documents[pdf_path] = semantic_chunk_text(text)

    return ingested_documents


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    documents = run_ingestion_pipeline()
    for document_path, chunks in documents.items():
        LOGGER.info("Ingested %s into %s chunks", document_path.name, len(chunks))
