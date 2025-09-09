# answer_gen.py — Groq (drop-in)
import os
import time
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from retriever import retrieve
from memory import load_history
from config import PDF_NAME

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment or .env")

client = Groq(api_key=API_KEY)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions strictly from the provided context. "
    "If the answer is not present, say you cannot find it in the document. "
    "Cite page numbers inline like (Page N) wherever you state a fact from the context. "
    "Be concise and factual; do not add emojis."
)

def _build_prompt(query: str, session_id: str, retrieved: List[Dict]) -> str:
    history = load_history(session_id, limit=8)
    history_text = "".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}\n"
        for role, text in history
    )
    ctx_blocks = []
    for r in retrieved:
        page = r.get("page", "?")
        content = r.get("content", "")
        ctx_blocks.append(f"(Page {page} from {PDF_NAME})\n{content}")
    context_text = "\n\n".join(ctx_blocks)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {query}\n\n"
        f"Answer clearly and cite pages like (Page N)."
    )

def _generate_with_backoff(prompt: str, attempts: int = 3, wait: float = 1.5) -> str:
    last_err = None
    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
            last_err = ValueError("Empty response")
        except Exception as e:
            last_err = e
        time.sleep(wait * (i + 1))
    raise last_err or RuntimeError("generation failed")

def answer_query(query: str, session_id: str, top_k: int = 5) -> Tuple[str, List[Dict], str]:
    retrieved = retrieve(query, top_k=top_k)
    prompt = _build_prompt(query, session_id, retrieved)

    try:
        answer = _generate_with_backoff(prompt)
        used = MODEL
    except Exception as e:
        answer = f"Generation error: {e}"
        used = MODEL

    citations = [{
        "page": r.get("page"),
        "pdf_url": r.get("pdf_url"),
        "pdf_name": r.get("pdf_name", PDF_NAME),
        "content": r.get("content", "")
    } for r in retrieved]

    return answer, citations, used
