# memory.py
import os
import json
from uuid import uuid4
from typing import List, Tuple

MEM_DIR = os.path.join("data", "memory")
os.makedirs(MEM_DIR, exist_ok=True)

def new_session_id() -> str:
    return uuid4().hex[:12]

def _path(sid: str) -> str:
    return os.path.join(MEM_DIR, f"{sid}.jsonl")

def save_message(session_id: str, role: str, text: str) -> None:
    rec = {"role": role, "text": text}
    with open(_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_history(session_id: str, limit: int = 50) -> List[Tuple[str, str]]:
    p = _path(session_id)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                rows.append((obj.get("role", ""), obj.get("text", "")))
            except Exception:
                continue
    return rows[-limit:]
