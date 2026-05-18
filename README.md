# Advanced RAG System

A production-style local Retrieval-Augmented Generation dashboard for private PDF question answering. The system ingests PDFs, chunks extracted text, stores local vector embeddings in ChromaDB, retrieves relevant context, and generates answers with Ollama.

## Features

- Streamlit dashboard for uploading PDFs and asking questions.
- Local PDF text extraction with `pypdf`.
- Token-aware chunking with `tiktoken`.
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent local vector storage with ChromaDB.
- Local answer generation with Ollama and `llama3.2`.
- No OpenAI API key required.

## Project Structure

```text
Advanced-RAG-System/
├── app.py
├── data/
├── requirements.txt
├── src/
│   ├── ingestion.py
│   ├── orchestration.py
│   └── vector_store.py
└── README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install Ollama from:

```text
https://ollama.com
```

Pull the local chat model:

```powershell
ollama pull llama3.2
```

## Run

Start the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal, usually:

```text
http://localhost:8501
```

## Workflow

1. Upload one or more PDF files from the sidebar.
2. Run the ingestion and embedding pipeline.
3. Ask a question in the main screen.
4. Review the generated answer and retrieved context.

## Notes

- Uploaded PDFs are stored locally in `data/`.
- ChromaDB files are stored locally in `chroma_db/`.
- Local generated files, vector databases, PDFs, virtual environments, and `.env` files are ignored by Git.
- If you change embedding models, delete `chroma_db/` and rerun ingestion so all vectors use the same embedding dimensions.
