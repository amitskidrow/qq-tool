from __future__ import annotations

from typing import Any, Dict, Optional

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from .qq_engine import Engine, DEFAULT_DB_URI
from .qq_embeddings import get_embedder
try:
    from .store_sqlite import Store
except Exception:
    Store = None  # type: ignore
try:
    from .rerank import rerank_pairs
except Exception:
    rerank_pairs = None  # type: ignore
try:
    from .store_sqlite import ingest_text as store_ingest_text  # type: ignore
except Exception:
    store_ingest_text = None  # type: ignore


class UpsertReq(BaseModel):
    id: str
    text: str
    meta: Optional[Dict[str, Any]] = None


class QueryReq(BaseModel):
    q: str
    k: int = 6
    alpha: float = 0.55
    rerank: bool = False


class SnapshotReq(BaseModel):
    to: str = Field(..., description="Absolute path to write the snapshot DB")


class IndexExportReq(BaseModel):
    out: str


class IndexImportReq(BaseModel):
    inp: str


class Query2Req(BaseModel):
    q: str
    k: int = 6
    alpha: float = 0.55
    rerank: bool = False
    graph_boost: bool = False


def build_app() -> FastAPI:
    app = FastAPI(title="qq API (UDS)", version="0.1.0", default_response_class=ORJSONResponse)

    @app.on_event("startup")
    def _startup() -> None:
        # Warm embedder and engine
        emb = get_embedder()
        _ = emb.encode(["warmup"])  # prime caches
        db_path = os.getenv("QQ_DB", DEFAULT_DB_URI)
        app.state.engine = Engine(embedder=emb, db_uri=db_path)
        # Initialize store for index admin and store-backed query
        try:
            if Store is not None:
                app.state.store = Store()
        except Exception:
            app.state.store = None

    @app.post("/upsert")
    def upsert(req: UpsertReq):
        try:
            eng: Engine = app.state.engine
            res = eng.upsert(req.id, req.text, req.meta)
            # Mirror into Store if available so /query2 can find docs
            st = getattr(app.state, "store", None)
            if st is not None and store_ingest_text is not None:
                try:
                    title = None
                    source = "api"
                    if req.meta:
                        title = req.meta.get("title") if isinstance(req.meta, dict) else None
                        source = req.meta.get("path") or source if isinstance(req.meta, dict) else source
                    store_ingest_text(
                        st,
                        doc_id=req.id,
                        text=req.text,
                        title=title,
                        kind="prose",
                        section_path=None,
                        source=source,
                        tags=None,
                        embed=True,
                    )
                except Exception:
                    # Do not fail the primary upsert if store ingest fails
                    pass
            return {"ok": True, **res}
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    def query(req: QueryReq):
        try:
            eng: Engine = app.state.engine
            hits, tm = eng.query(req.q, k=req.k, alpha=req.alpha, rerank=req.rerank)
            return {
                "results": [{"id": h.id, "score": h.score} for h in hits],
                "timings": tm.__dict__,
            }
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/snapshot")
    def snapshot(req: SnapshotReq):
        try:
            eng: Engine = app.state.engine
            res = eng.snapshot(req.to)
            return res
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/index/stats")
    def index_stats():
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        try:
            return st.stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/index/export")
    def index_export(req: IndexExportReq):
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        try:
            path = st.export_to(req.out)
            return {"ok": True, "out": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/index/import")
    def index_import(req: IndexImportReq):
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        try:
            st.import_from(req.inp)
            return {"ok": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query2")
    def query2(req: Query2Req):
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        try:
            pairs = st.search_hybrid(req.q, K_lex=req.k, K_sem=req.k, K_merge=req.k, alpha=req.alpha, graph_boost=req.graph_boost)
            ce_top = None
            if req.rerank and pairs and rerank_pairs is not None:
                mp = st.get_chunks([cid for cid, _ in pairs])
                passages = [mp.get(str(cid), {}).get("text", "") for cid, _ in pairs]
                try:
                    idx_scores = rerank_pairs("cross-encoder/ms-marco-MiniLM-L-6-v2", req.q, passages)
                    if idx_scores:
                        ce_top = float(idx_scores[0][1])
                        pairs = [(pairs[i][0], pairs[i][1]) for i, _ in idx_scores][: req.k]
                except Exception:
                    pass
            abstained = False
            S_THRESH = 0.12
            CE_THRESH = 0.20
            if pairs:
                if (pairs[0][1] < S_THRESH) and (ce_top is not None) and (ce_top < CE_THRESH):
                    abstained = True
            if abstained or not pairs:
                return {"answer": "No relevant paragraph found.", "abstained": True}
            top_id, top_score = pairs[0]
            src = st.get_chunks([top_id]).get(str(top_id)) or {}
            ans = (src.get("text") or "").split()
            if len(ans) > 120:
                ans = ans[:120]
            answer = " ".join(ans)
            return {
                "answer": answer,
                "source": {
                    "chunk_id": top_id,
                    "doc_id": src.get("doc_id"),
                    "section_path": src.get("section_path"),
                },
                "scores": {"hybrid": float(top_score), "ce": ce_top},
                "policy": "extractive_only",
                "abstained": False,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/info")
    def info():
        """Return embedding/backend diagnostics from the server process."""
        import onnxruntime as ort
        data = {
            "env": {
                "QQ_EMBED_MODEL": os.getenv("QQ_EMBED_MODEL"),
                "QQ_ORT_PROVIDER": os.getenv("QQ_ORT_PROVIDER"),
                "QQ_ORT_PROVIDERS": os.getenv("QQ_ORT_PROVIDERS"),
            },
            "ort": {
                "available_providers": list(ort.get_available_providers()),
                "version": getattr(ort, "__version__", None),
            },
            "embedding": {},
            "vector": {},
        }
        try:
            emb = get_embedder()
            _ = emb.encode(["warmup"])  # prime
            info_obj = getattr(emb, "info", None)
            if info_obj is not None:
                data["embedding"] = {
                    "model": getattr(info_obj, "model_name", None),
                    "backend": getattr(info_obj, "backend", None),
                    "dim": getattr(info_obj, "dim", None),
                    "warmed": getattr(info_obj, "warmed", None),
                    "load_ms": getattr(info_obj, "load_ms", None),
                    "providers": getattr(info_obj, "providers", None),
                }
        except Exception as e:
            data["embedding"] = {"error": str(e)}
        # sqlite-vec availability
        try:
            import sqlite_vec  # type: ignore

            data["vector"]["sqlite_vec"] = True
            data["vector"]["sqlite_vec_version"] = getattr(sqlite_vec, "__version__", None)
        except Exception:
            data["vector"]["sqlite_vec"] = False
        # store stats if available
        st = getattr(app.state, "store", None)
        if st is not None:
            try:
                data["store_stats"] = st.stats()
            except Exception:
                pass
        return data

    return app


app = build_app()
