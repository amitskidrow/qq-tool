from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .embeddings import embed_texts
from .store_sqlite import SqliteVecStore
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
        db_path = cfg.get("database", {}).get("path")
        if not db_path:
            raise HTTPException(status_code=500, detail="database.path not configured")
        store = SqliteVecStore(db_path)
        try:
            q_vec = embed_texts([q])[0]
            ns_final = ns or infer_namespace()
            results = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk)
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
