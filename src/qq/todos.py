from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_HOME
from .util import now_ist


def _todo_path() -> Path:
    d = DEFAULT_HOME / "todos"
    d.mkdir(parents=True, exist_ok=True)
    return d / "todos.json"


def load_todos() -> List[Dict]:
    p = _todo_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return []
    return []


def save_todos(items: List[Dict]) -> None:
    _todo_path().write_text(json.dumps(items, ensure_ascii=False, indent=2))


def add_todo(text: str, session_id: str | None = None, ns: str | None = None) -> str:
    items = load_todos()
    tid = str(uuid.uuid4())
    items.append({"id": tid, "text": text, "session": session_id, "ns": ns, "created_at": now_ist().isoformat(), "resolved": False, "answer": None})
    save_todos(items)
    return tid


def resolve_todo(tid: str, answer: str) -> bool:
    items = load_todos()
    for it in items:
        if it.get("id") == tid:
            it["resolved"] = True
            it["answer"] = answer
            save_todos(items)
            return True
    return False

