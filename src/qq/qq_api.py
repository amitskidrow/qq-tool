from __future__ import annotations

from typing import Any, Dict, Optional

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse, StreamingResponse
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

    @app.get("/index/list")
    def index_list(
        q: Optional[str] = Query(default=None),
        like: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        """
        List documents with optional FTS search over chunks.

        Returns rows with: id, uri, title, size, updated, score, chunk_count, token_count, meta_summary.
        """
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        try:
            conn = st.conn  # use underlying sqlite connection
            rows: list[dict] = []
            if q:
                # Build safe FTS expression similar to Store._safe_fts_query
                try:
                    expr = st._safe_fts_query(q)  # type: ignore[attr-defined]
                except Exception:
                    expr = None
                if not expr:
                    return {"rows": []}
                sql = (
                    """
                    SELECT d.doc_id AS id,
                           d.source AS uri,
                           d.title AS title,
                           d.updated_at AS updated,
                           (SELECT SUM(length(c2.content)) FROM chunks c2 WHERE c2.doc_id=d.doc_id) AS size,
                           MIN(bm25(fts)) AS bm25,
                           (SELECT COUNT(1) FROM chunks c3 WHERE c3.doc_id=d.doc_id) AS chunk_count,
                           (SELECT COALESCE(SUM(c4.token_count),0) FROM chunks c4 WHERE c4.doc_id=d.doc_id) AS token_count,
                           substr(d.tags, 1, 120) AS meta_summary
                    FROM fts
                    JOIN chunks c ON c.chunk_id = fts.chunk_id
                    JOIN docs d ON d.doc_id = c.doc_id
                    WHERE fts MATCH ?
                    GROUP BY d.doc_id
                    ORDER BY bm25 ASC
                    LIMIT ? OFFSET ?
                    """
                )
                cur = conn.execute(sql, (expr, int(limit), int(offset)))
                tmp = cur.fetchall()
                if not tmp:
                    return {"rows": []}
                # Normalize bm25 to score in [0,1] over this page
                bm25_vals = [float(r["bm25"]) if "bm25" in r.keys() else float(r[5]) for r in tmp]
                max_r = max(bm25_vals)
                min_r = min(bm25_vals)
                denom = max(max_r - min_r, 1e-9)
                for r in tmp:
                    bm25_val = float(r["bm25"]) if "bm25" in r.keys() else float(r[5])
                    score = 1.0 - ((bm25_val - min_r) / denom)
                    rows.append(
                        {
                            "id": r["id"],
                            "uri": r["uri"],
                            "title": r["title"],
                            "size": int(r["size"]) if r["size"] is not None else 0,
                            "updated": r["updated"],
                            "score": float(score),
                            "chunk_count": int(r["chunk_count"]) if r["chunk_count"] is not None else 0,
                            "token_count": int(r["token_count"]) if r["token_count"] is not None else 0,
                            "meta_summary": r["meta_summary"],
                        }
                    )
                return {"rows": rows}
            # No search: optional LIKE filter on source/title; else recent
            where = []
            params: list[object] = []
            if like:
                where.append("(d.source LIKE ? OR d.title LIKE ?)")
                pat = like
                params.extend([pat, pat])
            sql = (
                """
                SELECT d.doc_id AS id,
                       d.source AS uri,
                       d.title AS title,
                       d.updated_at AS updated,
                       (SELECT SUM(length(c2.content)) FROM chunks c2 WHERE c2.doc_id=d.doc_id) AS size,
                       (SELECT COUNT(1) FROM chunks c3 WHERE c3.doc_id=d.doc_id) AS chunk_count,
                       (SELECT COALESCE(SUM(c4.token_count),0) FROM chunks c4 WHERE c4.doc_id=d.doc_id) AS token_count,
                       substr(d.tags, 1, 120) AS meta_summary
                FROM docs d
                {where}
                ORDER BY d.updated_at DESC
                LIMIT ? OFFSET ?
                """
            ).format(where=("WHERE " + " AND ".join(where)) if where else "")
            params.extend([int(limit), int(offset)])
            cur = conn.execute(sql, params)
            for r in cur.fetchall():
                rows.append(
                    {
                        "id": r["id"],
                        "uri": r["uri"],
                        "title": r["title"],
                        "size": int(r["size"]) if r["size"] is not None else 0,
                        "updated": r["updated"],
                        "score": 0.0,
                        "chunk_count": int(r["chunk_count"]) if r["chunk_count"] is not None else 0,
                        "token_count": int(r["token_count"]) if r["token_count"] is not None else 0,
                        "meta_summary": r["meta_summary"],
                    }
                )
            return {"rows": rows}
        except HTTPException:
            raise
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

    @app.get("/index/get")
    def index_get(id: Optional[str] = Query(default=None), uri: Optional[str] = Query(default=None)):
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")
        if not id and not uri:
            raise HTTPException(status_code=400, detail="id or uri required")
        try:
            conn = st.conn
            if id:
                row = conn.execute(
                    "SELECT doc_id, title, source, tags FROM docs WHERE doc_id=?",
                    (id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT doc_id, title, source, tags FROM docs WHERE source=?",
                    (uri,),
                ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="document not found")
            doc_id = row["doc_id"] if "doc_id" in row.keys() else row[0]
            title = row["title"] if "title" in row.keys() else row[1]
            source = row["source"] if "source" in row.keys() else row[2]
            tags = row["tags"] if "tags" in row.keys() else row[3]
            # Fetch full text by concatenating chunks in order
            cur = conn.execute(
                "SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_id",
                (doc_id,),
            )
            parts = [r[0] for r in cur.fetchall()]
            text = "\n\n".join(p for p in parts if p)
            meta: Dict[str, Any]
            try:
                meta = json.loads(tags) if isinstance(tags, str) and tags else {}
            except Exception:
                meta = {}
            return {"id": doc_id, "uri": source, "title": title, "meta": meta, "text": text}
        except HTTPException:
            raise
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

    @app.get("/export")
    def export(ids: Optional[str] = Query(default=None), q: Optional[str] = Query(default=None), limit: int = Query(default=500, ge=1, le=5000)):
        """
        Stream JSONL export of documents. If `ids` (comma-separated) is provided, it takes precedence.
        Otherwise when `q` is provided, use FTS search to select up to `limit` docs.
        Without both, stream recent docs up to `limit`.
        """
        st = getattr(app.state, "store", None)
        if st is None:
            raise HTTPException(status_code=500, detail="store not initialized")

        try:
            conn = st.conn
            doc_ids: list[str] = []
            if ids:
                doc_ids = [x.strip() for x in ids.split(",") if x.strip()]
            elif q:
                try:
                    expr = st._safe_fts_query(q)  # type: ignore[attr-defined]
                except Exception:
                    expr = None
                if expr:
                    sql = (
                        """
                        SELECT d.doc_id AS id, MIN(bm25(fts)) AS bm25
                        FROM fts
                        JOIN chunks c ON c.chunk_id = fts.chunk_id
                        JOIN docs d ON d.doc_id = c.doc_id
                        WHERE fts MATCH ?
                        GROUP BY d.doc_id
                        ORDER BY bm25 ASC
                        LIMIT ?
                        """
                    )
                    cur = conn.execute(sql, (expr, int(limit)))
                    doc_ids = [r["id"] if "id" in r.keys() else r[0] for r in cur.fetchall()]
            else:
                cur = conn.execute(
                    "SELECT doc_id FROM docs ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                )
                doc_ids = [r["doc_id"] if "doc_id" in r.keys() else r[0] for r in cur.fetchall()]

            def _iter() -> Any:
                for did in doc_ids:
                    row = conn.execute(
                        "SELECT doc_id, title, source, tags FROM docs WHERE doc_id=?",
                        (did,),
                    ).fetchone()
                    if not row:
                        continue
                    ddoc_id = row["doc_id"] if "doc_id" in row.keys() else row[0]
                    title = row["title"] if "title" in row.keys() else row[1]
                    source = row["source"] if "source" in row.keys() else row[2]
                    tags = row["tags"] if "tags" in row.keys() else row[3]
                    cur2 = conn.execute(
                        "SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_id",
                        (ddoc_id,),
                    )
                    parts = [r2[0] for r2 in cur2.fetchall()]
                    text = "\n\n".join(p for p in parts if p)
                    try:
                        meta = json.loads(tags) if isinstance(tags, str) and tags else {}
                    except Exception:
                        meta = {}
                    obj = {
                        "id": ddoc_id,
                        "uri": source,
                        "title": title,
                        "meta": meta,
                        "text": text,
                    }
                    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                    yield line

            return StreamingResponse(_iter(), media_type="application/x-ndjson")
        except HTTPException:
            raise
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
