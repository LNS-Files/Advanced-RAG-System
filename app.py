from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.ingestion import extract_text_from_pdf, semantic_chunk_text
from src.orchestration import generate_answer, retrieve_context
from src.vector_store import add_chunks_to_db, initialize_vector_db


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ALLOWED_FILE_TYPE = "pdf"


def configure_page() -> None:
    st.set_page_config(
        page_title="Production Advanced RAG Dashboard",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Production Advanced RAG Dashboard")


def save_uploaded_pdfs(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            LOGGER.warning("Skipped non-PDF upload: %s", uploaded_file.name)
            continue

        destination = DATA_DIR / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def run_ingestion_and_embedding() -> tuple[int, int]:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the data directory.")

    collection = initialize_vector_db()
    processed_files = 0
    stored_chunks = 0

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            LOGGER.warning("No extractable text found in %s", pdf_path)
            continue

        chunks = semantic_chunk_text(text)
        if not chunks:
            LOGGER.warning("No chunks generated for %s", pdf_path)
            continue

        stored_ids = add_chunks_to_db(
            chunks=chunks,
            collection=collection,
            source=pdf_path.name,
        )
        processed_files += 1
        stored_chunks += len(stored_ids)

    return processed_files, stored_chunks


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Document Pipeline")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=[ALLOWED_FILE_TYPE],
            accept_multiple_files=True,
        )

        if uploaded_files:
            saved_paths = save_uploaded_pdfs(uploaded_files)
            if saved_paths:
                st.success(f"Saved {len(saved_paths)} PDF file(s) to data/.")
            else:
                st.warning("No valid PDF files were saved.")

        if st.button("Run Ingestion & Embedding Pipeline", type="primary"):
            try:
                with st.spinner("Extracting PDFs, chunking text, and generating local embeddings..."):
                    processed_files, stored_chunks = run_ingestion_and_embedding()
                st.success(
                    f"Pipeline complete: processed {processed_files} file(s), "
                    f"stored {stored_chunks} chunk(s)."
                )
            except FileNotFoundError as exc:
                st.warning(str(exc))
            except Exception as exc:
                LOGGER.exception("Ingestion pipeline failed")
                st.error(f"Ingestion pipeline failed: {exc}")


def render_main_screen() -> None:
    st.subheader("Ask Your Knowledge Base")
    user_query = st.text_input(
        "Question",
        placeholder="Ask a question about the ingested PDFs...",
    )

    if st.button("Generate Answer", type="primary"):
        if not user_query.strip():
            st.warning("Enter a question before generating an answer.")
            return

        try:
            with st.spinner("Retrieving context and generating answer..."):
                context = retrieve_context(user_query)
                answer = generate_answer(user_query, context)

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Retrieved Context"):
                st.write(context or "No relevant context was retrieved.")
        except Exception as exc:
            LOGGER.exception("Answer generation failed")
            st.error(f"Answer generation failed: {exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    configure_page()
    render_sidebar()
    render_main_screen()


if __name__ == "__main__":
    main()
