from __future__ import annotations

import json
from typing import Dict, List

from .config import load_config
from .answer_contract import ANSWER_CONTRACT_SYSTEM, build_user_prompt
from .embeddings import embed_texts
from .util import infer_namespace
from .store_sqlite import SqliteVecStore


def vector_retrieve(store: SqliteVecStore, q: str, ns: str, topk: int = 8):
    q_vec = embed_texts([q])[0]
    res = store.search_dense(q_vec, namespace=ns, topk=topk)
    contexts = []
    for r in res:
        contexts.append({
            "id": r.id,
            "score": float(r.score),
            "source": r.uri,
            "text": r.text or "",
        })
    return contexts


def answer(store: SqliteVecStore, provider: str, model: str, question: str, ns: str | None = None) -> Dict:
    cfg = load_config()
    ns = ns or infer_namespace()
    ctx = vector_retrieve(store, question, ns, topk=10)
    prompt = build_user_prompt(question, ctx)
    sys_prompt = cfg.get("client_model", {}).get("system_prompt") or ANSWER_CONTRACT_SYSTEM
    if provider == "openai":
        from .models.openai_client import complete_json
    else:
        from .models.gemini_client import complete_json
    out = complete_json(model, sys_prompt, prompt)
    # ensure citations filled if empty
    if not out.get("citations"):
        out["citations"] = [{"id": c["id"], "source": c["source"], "score": c["score"]} for c in ctx[:5]]
    return out
