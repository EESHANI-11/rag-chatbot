import os
import uuid
from typing import Iterable, Tuple

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

from config import PDF_LOCAL_PATH, CHROMA_DIR, CHROMA_COLLECTION, PDF_NAME, PDF_URL

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_pdf_pages(path: str) -> Iterable[Tuple[int, str]]:
    doc = fitz.open(path)
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        yield i + 1, text or ""


def chunk_text(s: str, chunk_size: int = 1200, overlap: int = 150) -> Iterable[str]:
    s = " ".join(s.split())
    if not s:
        return
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + chunk_size)
        yield s[start:end].strip()
        if end == n:
            break
        start = max(0, end - overlap)


def main():
    if not os.path.exists(PDF_LOCAL_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_LOCAL_PATH}")

    pages = list(read_pdf_pages(PDF_LOCAL_PATH))
    print(f"Loaded {len(pages)} pages from {os.path.basename(PDF_LOCAL_PATH)}")

    # Build chunks with metadata
    items = []
    for page_no, text in pages:
        for ch in chunk_text(text):
            items.append({
                "id": str(uuid.uuid4()),
                "content": ch,
                "page": page_no,
                "pdf_name": PDF_NAME,
                "pdf_url": PDF_URL,
            })

    if not items:
        print("No text found to index.")
        return

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION)

    embedder = SentenceTransformer(EMB_MODEL_NAME)

    # Insert in batches
    B = 100
    for i in range(0, len(items), B):
        batch = items[i : i + B]
        embeddings = embedder.encode([b["content"] for b in batch], normalize_embeddings=True).tolist()
        collection.add(
            ids=[b["id"] for b in batch],
            documents=[b["content"] for b in batch],
            embeddings=embeddings,
            metadatas=[{"page": b["page"], "pdf_name": b["pdf_name"], "pdf_url": b["pdf_url"]} for b in batch],
        )

    print(f"Done. Indexed {len(items)} chunks into '{CHROMA_COLLECTION}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
