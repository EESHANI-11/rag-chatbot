import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, CHROMA_COLLECTION, PDF_NAME, PDF_URL

_EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_CLIENT = chromadb.PersistentClient(path=CHROMA_DIR)
_COL = _CLIENT.get_collection(CHROMA_COLLECTION)


def retrieve(query: str, top_k: int = 5):
    qvec = _EMB.encode([query], normalize_embeddings=True).tolist()
    res = _COL.query(query_embeddings=qvec, n_results=top_k)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "content": res["documents"][0][i],
            "page": int(res["metadatas"][0][i].get("page", 0)),
            "pdf_name": res["metadatas"][0][i].get("pdf_name", PDF_NAME),
            "pdf_url": res["metadatas"][0][i].get("pdf_url", PDF_URL),
        })
    return out
