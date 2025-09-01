from __future__ import annotations

import json
from typing import Dict, List

from .config import load_config
from .answer_contract import ANSWER_CONTRACT_SYSTEM, build_user_prompt
from .embeddings import embed_texts
from .util import infer_namespace


def vector_retrieve(client, collection: str, q: str, ns: str, topk: int = 8):
    from qdrant_client.http.models import Filter, FieldCondition, Match

    q_vec = embed_texts([q])[0]
    filt = Filter(must=[FieldCondition(key="namespace", match=Match(value=ns))])
    res = client.search(collection_name=collection, query_vector=q_vec.tolist(), limit=topk, query_filter=filt)
    contexts = []
    for p in res:
        pl = p.payload or {}
        contexts.append({
            "id": p.id,
            "score": float(p.score or 0.0),
            "source": pl.get("source"),
            "text": pl.get("text") or "",
        })
    return contexts


def answer(client, collection: str, provider: str, model: str, question: str, ns: str | None = None) -> Dict:
    cfg = load_config()
    ns = ns or infer_namespace()
    ctx = vector_retrieve(client, collection, question, ns, topk=10)
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

