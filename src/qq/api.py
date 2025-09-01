from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .vector_store import connect as vs_connect
from .embeddings import embed_texts
from .hybrid import bm25_rank, hybrid_merge
from .util import infer_namespace


def build_app() -> FastAPI:
    app = FastAPI(title="qq API", version="0.0.1")

    @app.get("/health")
    def health():
        return {"ok": True}

    # Alias for readiness probes
    @app.get("/ready")
    def ready():
        return {"ok": True}

    @app.get("/query")
    def api_query(q: str, ns: str | None = None, topk: int = 5):
        cfg = load_config()
        qdrant_cfg = cfg.get("vector_store", {})
        qdrant_url = qdrant_cfg.get("url", "http://localhost:6333")
        collection = qdrant_cfg.get("collection", "qq_data")
        api_key = qdrant_cfg.get("api_key")
        client = vs_connect(qdrant_url, api_key)
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            q_vec = embed_texts([q])[0]
            ns_final = ns or infer_namespace()
            filt = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=ns_final))])
            res = client.search(collection_name=collection, query_vector=q_vec.tolist(), limit=topk, query_filter=filt)
            out = []
            for p in res:
                pl = p.payload or {}
                out.append({"id": p.id, "score": float(p.score or 0.0), "source": pl.get("source"), "namespace": pl.get("namespace"), "snippet": (pl.get("text") or "")[:240]})
            return {"results": out, "ns": ns_final}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
