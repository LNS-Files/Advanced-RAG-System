from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.ingestion import chunk_pages, extract_pages
from src.orchestration import retrieve_context, rewrite_query, stream_answer
from src.vector_store import add_chunks_to_db, initialize_vector_db


@st.cache_resource
def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


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
    embedding_model = _load_embedding_model()
    processed_files = 0
    stored_chunks = 0

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        if not pages:
            LOGGER.warning("No extractable text found in %s", pdf_path)
            continue

        page_chunks = chunk_pages(pages)
        if not page_chunks:
            LOGGER.warning("No chunks generated for %s", pdf_path)
            continue

        texts = [c["text"] for c in page_chunks]
        per_chunk_metadata = [{"page": c["page"]} for c in page_chunks]
        stored_ids = add_chunks_to_db(
            chunks=texts,
            collection=collection,
            embedding_model=embedding_model,
            source=pdf_path.name,
            per_chunk_metadata=per_chunk_metadata,
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

        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def render_main_screen() -> None:
    st.subheader("Chat with Your Knowledge Base")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_query := st.chat_input("Ask a question about the ingested PDFs..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            try:
                history = st.session_state.messages[:-1]
                with st.spinner("Rewriting query..."):
                    search_query = rewrite_query(user_query, history=history)
                with st.spinner("Retrieving context..."):
                    context, citations = retrieve_context(search_query)
                answer = st.write_stream(
                    stream_answer(user_query, context, history=history)
                )
            except Exception as exc:
                LOGGER.exception("Answer generation failed")
                answer = f"Answer generation failed: {exc}"
                context, citations = "", []
                search_query = user_query
                st.write(answer)

            if citations:
                citation_lines = []
                for c in citations:
                    label = c["source"]
                    if c["page"] is not None:
                        label += f", page {c['page']}"
                    citation_lines.append(f"- {label}")
                st.markdown("**Sources:**\n" + "\n".join(citation_lines))

            with st.expander("Retrieved Context"):
                if search_query != user_query:
                    st.caption(f"Searched for: _{search_query}_")
                st.write(context or "No relevant context was retrieved.")

        st.session_state.messages.append({"role": "assistant", "content": answer})


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
