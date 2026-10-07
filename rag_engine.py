from pathlib import Path

import chromadb
import time
from docx import Document
from pypdf import PdfReader

from config import (
    DOCUMENTS_DIR,
    VECTOR_DB_DIR,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CONTEXT_CHARS,
)
from gemini_service import embed_text, generate_answer


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# Persistent Chroma database.
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

collection = chroma_client.get_or_create_collection(
    name="lawyer_corruption_documents",
    metadata={"hnsw:space": "cosine"},
)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                f"[Page {page_number}]\n{text}"
            )

    return "\n\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_document(path: Path) -> str:
    """Read PDF, DOCX or TXT."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    if suffix == ".txt":
        return read_txt(path)

    return ""


def chunk_text(text: str):
    """
    Very simple beginner-friendly word chunking.

    Example:
    chunk_size=700 and overlap=100 means:
    chunk 1 -> words 0..699
    chunk 2 -> words 600..1299
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


def clear_collection():
    """Delete existing Chroma data before a fresh ingest."""
    current = collection.get()

    ids = current.get("ids", [])

    if ids:
        collection.delete(ids=ids)


def ingest_documents():
    """
    Read documents -> split into chunks -> Gemini embeddings -> Chroma.
    """
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    files_processed = 0
    chunks_created = 0

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = read_document(path).strip()

        if not text:
            continue

        files_processed += 1

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = f"{path.name}__chunk_{chunk_number}"

            vector = embed_text(
                chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )

            time.sleep(1)

            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(vector)

            metadatas.append(
                {
                    "source": path.name,
                    "chunk_number": chunk_number,
                }
            )

            chunks_created += 1

    if not chunks_created:
        raise ValueError(
            "No supported documents found. Put PDF, DOCX or TXT files "
            "inside the documents folder, then call POST /ingest."
        )

    clear_collection()

    # Add all chunks to Chroma.
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "files_processed": files_processed,
        "chunks_created": chunks_created,
    }


def retrieve_context(question: str, top_k: int = TOP_K):
    """Find the most relevant chunks from Chroma."""

    if collection.count() == 0:
        raise ValueError(
            "Vector database is empty. Call POST /ingest first."
        )

    query_vector = embed_text(
        question,
        task_type="RETRIEVAL_QUERY"
    )

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items = []

    for index, doc in enumerate(docs):
        distance = float(distances[index])

        similarity = max(
            0.0,
            min(1.0, 1.0 - distance)
        )

        items.append(
            {
                "text": doc,
                "source": metas[index].get(
                    "source",
                    "unknown"
                ),
                "chunk_number": metas[index].get(
                    "chunk_number",
                    -1
                ),
                "similarity": round(
                    similarity,
                    4
                ),
            }
        )

    return items

def build_prompt(question: str, retrieved_items):
    """Build a small prompt using only relevant context."""

    context_parts = []
    total_chars = 0

    for item in retrieved_items:
        text = item["text"]

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        text = text[:remaining]

        context_parts.append(text)
        total_chars += len(text)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a legal RAG assistant for lawyer corruption cases.

Use ONLY the CONTEXT to answer the QUESTION.

Rules:
- Exactly 5 short points.
- One important fact per point.
- Simple English.
- No guessing.
- No repetition.
- No legal advice.
- Missing information: "Not available in the document."
- Output only 5 points.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    return prompt

def ask_question(question: str):
    """Complete RAG flow: retrieve -> prompt -> Gemini."""
    retrieved = retrieve_context(question)

    prompt = build_prompt(question, retrieved)

    generation = generate_answer(prompt)

    best_similarity = (
        retrieved[0]["similarity"] if retrieved else 0.0
    )

    return {
        "question": question,
        "answer": generation["answer"],
        "sources": retrieved,
        "retrieval_similarity": best_similarity,
        "usage": {
            "latency_ms": generation["latency_ms"],
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "total_tokens": generation["total_tokens"],
            "estimated_api_cost_usd": generation["estimated_api_cost_usd"],
        },
    }


def list_documents():
    """Return document filenames placed inside documents/."""
    result = []

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append(path.name)

    return result
