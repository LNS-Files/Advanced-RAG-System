from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import tiktoken
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, ENCODING_NAME

LOGGER = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs for every readable page in the PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        reader = PdfReader(path)
    except (PdfReadError, OSError, ValueError) as exc:
        LOGGER.warning("Skipping unreadable PDF %s: %s", path, exc)
        return []

    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (PdfReadError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping page %d in %s: %s", page_number, path, exc)
            continue
        if text.strip():
            pages.append((page_number, text))

    return pages


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF (all pages joined)."""
    return "\n\n".join(text for _, text in extract_pages(pdf_path))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    encoding_name: str = ENCODING_NAME,
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
        if not chunk_tokens:
            break
        chunks.append(encoding.decode(chunk_tokens))

    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk page-level text and preserve the source page number per chunk."""
    result: list[dict] = []
    for page_num, text in pages:
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            result.append({"text": chunk, "page": page_num})
    return result


def iter_pdf_files(data_dir: str | Path = DATA_DIR) -> Iterable[Path]:
    path = Path(data_dir)
    if not path.exists():
        LOGGER.warning("Data directory does not exist: %s", path)
        return ()
    return sorted(path.glob("*.pdf"))


def run_ingestion_pipeline(data_dir: str | Path = DATA_DIR) -> dict[Path, list[dict]]:
    """Return {pdf_path: [{"text": str, "page": int}, ...]} for every PDF."""
    ingested: dict[Path, list[dict]] = {}
    for pdf_path in iter_pdf_files(data_dir):
        pages = extract_pages(pdf_path)
        ingested[pdf_path] = chunk_pages(pages) if pages else []
    return ingested


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    documents = run_ingestion_pipeline()
    for doc_path, chunks in documents.items():
        LOGGER.info("Ingested %s into %d chunks", doc_path.name, len(chunks))
