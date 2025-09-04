from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .embeddings import embed_texts, embedding_dim
from .store_typesense import TypesenseStore
from .util import infer_namespace


def build_app() -> FastAPI:
    app = FastAPI(title="qq API", version="0.0.1")

    # Warm the embedding model and prepare Typesense on startup to avoid first-request latency
    @app.on_event("startup")
    def _warmup():
        try:
            # Load model into memory and infer embedding dim
            _ = embed_texts(["warmup"])
            dim = embedding_dim()
        except Exception:
            dim = None

        # Reuse a single Typesense client across requests
        try:
            app.state.store = TypesenseStore()
            if dim is not None:
                app.state.store.ensure_schema(dim)
        except Exception:
            # Defer failures to request-time exceptions
            app.state.store = TypesenseStore()

    @app.get("/health")
    def health():
        return {"ok": True}

    # Alias for readiness probes
    @app.get("/ready")
    def ready():
        return {"ok": True}

    @app.get("/query")
    def api_query(q: str, ns: str | None = None, topk: int = 5, hybrid: bool = True):
        store: TypesenseStore = getattr(app.state, "store", TypesenseStore())
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

# Export module-level app for Uvicorn discovery (uvicorn qq.api:app)
app = build_app()
