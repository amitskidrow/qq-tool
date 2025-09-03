from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .embeddings import embed_texts
from .store_typesense import TypesenseStore
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
    def api_query(q: str, ns: str | None = None, topk: int = 5, hybrid: bool = True):
        store = TypesenseStore()
        try:
            q_vec = embed_texts([q])[0]
            ns_final = ns or infer_namespace()
            if hybrid:
                results = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk)
            else:
                results = store.search_dense(q_vec, namespace=ns_final, topk=topk)
            out = []
            for r in results:
                out.append({
                    "id": r.id,
                    "score": r.score,
                    "source": r.uri,
                    "namespace": r.namespace,
                    "snippet": (r.text or "")[:240],
                })
            return {"results": out, "ns": ns_final}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
