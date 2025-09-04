from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from .qq_engine import Engine
from .qq_embeddings import get_embedder


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


def build_app() -> FastAPI:
    app = FastAPI(title="qq API (UDS)", version="0.1.0", default_response_class=ORJSONResponse)

    @app.on_event("startup")
    def _startup() -> None:
        # Warm embedder and engine
        emb = get_embedder()
        _ = emb.encode(["warmup"])  # prime caches
        app.state.engine = Engine(embedder=emb)

    @app.post("/upsert")
    def upsert(req: UpsertReq):
        try:
            eng: Engine = app.state.engine
            res = eng.upsert(req.id, req.text, req.meta)
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

    return app


app = build_app()

