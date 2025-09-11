from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx
import time
import typer

app = typer.Typer(add_completion=False, help="qq CLI (UDS client)")
index_app = typer.Typer(help="Index admin commands (new-arch store)")


def _uds_path() -> str:
    # Global default UDS path
    return os.getenv("QQ_UDS", os.path.expanduser("~/.qq/qq.sock"))


def _client() -> httpx.Client:
    transport = httpx.HTTPTransport(uds=_uds_path())
    # Base URL host/path is ignored with UDS, but required by httpx
    return httpx.Client(transport=transport, base_url="http://qq.local", timeout=30.0)


# In-process fallback
def _fallback_engine():
    from .qq_engine import Engine, DEFAULT_DB_URI

    db = os.getenv("QQ_DB", DEFAULT_DB_URI)
    # Do not instantiate embedder here; Engine will decide based on vec availability
    return Engine(db_uri=db)


def _hash_id(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8"))
    return h.hexdigest()


def _iter_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
    else:
        skip_dirs = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}
        for f in p.rglob("*"):
            parts = set(f.parts)
            if parts & skip_dirs:
                continue
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".rst", ".log"}:
                yield f


@app.command()
def ingest(path: str = typer.Argument(..., help="File or directory to ingest")):
    p = Path(path)
    if not p.exists():
        typer.echo(json.dumps({"ok": False, "error": f"path not found: {path}"}))
        raise typer.Exit(2)
    added = 0
    try:
        c = _client()
        for f in _iter_files(p):
            text = f.read_text(encoding="utf-8", errors="ignore")
            doc_id = _hash_id(f)
            payload = {"id": doc_id, "text": text, "meta": {"path": str(f)}}
            r = c.post("/upsert", json=payload)
            r.raise_for_status()
            added += 1
        typer.echo(json.dumps({"ok": True, "ingested": added, "mode": "server"}))
    except Exception:
        # Fallback to new-arch store
        try:
            from .store_sqlite import Store, ingest_text
        except Exception as e:
            # If store import fails, fallback to legacy engine
            eng = _fallback_engine()
            for f in _iter_files(p):
                text = f.read_text(encoding="utf-8", errors="ignore")
                doc_id = _hash_id(f)
                eng.upsert(doc_id, text, {"path": str(f)})
                added += 1
            typer.echo(json.dumps({"ok": True, "ingested": added, "mode": "local-legacy"}))
            return
        store = Store()
        totals = {"docs": 0, "chunks": 0, "duplicates": 0, "embedded": 0}
        for f in _iter_files(p):
            text = f.read_text(encoding="utf-8", errors="ignore")
            doc_id = _hash_id(f)
            counts = ingest_text(
                store,
                doc_id=doc_id,
                text=text,
                title=f.name,
                kind="prose",
                section_path=None,
                source=str(f),
                tags=None,
                embed=True,
            )
            totals["docs"] += counts.docs
            totals["chunks"] += counts.chunks
            totals["duplicates"] += counts.duplicates
            totals["embedded"] += counts.embedded
            added += 1
        out = {"ok": True, "ingested": added, "mode": "local-store", **totals}
        typer.echo(json.dumps(out))


def _render_plain(
    q: str,
    k: int,
    mode: str,
    db: str,
    results: List[Tuple[str, float]],
    elapsed_ms: float,
    texts: Optional[dict] = None,
) -> str:
    from qq import __version__
    header = f"QQ {__version__} | mode={mode} | k={k} | db={db} | q=\"{q}\" | {int(elapsed_ms)}ms"
    lines = [header]
    for i, (rid, score) in enumerate(results, start=1):
        # Always show id/score
        lines.append(f"{i} | score={score:.4f} | id={rid}")
        # If text content available (local mode), print a plain-text body block
        if texts is not None and rid in texts:
            body = texts[rid].get("text", "")
            path = texts[rid].get("path") or ""
            if path:
                lines.append(f"path={path}")
            # Trim excessively long output to keep terminal usable
            max_chars = int(os.getenv("QQ_MAX_PRINT_CHARS", "4000"))
            if max_chars > 0 and len(body) > max_chars:
                lines.append(body[:max_chars])
                lines.append(f"[... truncated {len(body) - max_chars} chars ...]")
            else:
                lines.append(body)
    lines.append(f"results={len(results)} | elapsed={int(elapsed_ms)}ms")
    return "\n".join(lines)


@app.command()
def query(
    q: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(5, "--k", help="Top-K results"),
    alpha: float = typer.Option(0.6, "--alpha", help="Dense weight [0,1] (hybrid only)"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranker"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    db = os.getenv("QQ_DB", os.path.expanduser("~/.qq/global.db"))
    # Try server first
    try:
        c = _client()
        r = c.post("/query", json={"q": q, "k": k, "alpha": alpha, "rerank": rerank})
        r.raise_for_status()
        data = r.json()
        if json_out:
            typer.echo(json.dumps(data))
            return
        results = [(h.get("id"), float(h.get("score", 0.0))) for h in data.get("results", [])]
        elapsed = float(data.get("timings", {}).get("total_ms", 0.0))
        # Mode unknown from server; approximate
        mode = "hybrid"
        typer.echo(_render_plain(q, k, mode, db, results, elapsed))
        return
    except Exception:
        pass

    # Prefer new-arch store fallback; if not available, use legacy engine
    try:
        from .store_sqlite import Store
        t0 = time.perf_counter()
        store = Store()
        # Use k for all caps to keep CLI expectations simple
        pairs = store.search_hybrid(q, K_lex=k, K_sem=k, K_merge=k, alpha=alpha)
        t1 = time.perf_counter()
        # Convert to (id, score) with id as chunk_id string
        results = [(str(cid), float(score)) for cid, score in pairs]
        text_map = {}
        if results:
            chunk_ids = [int(rid) for rid, _ in results]
            text_map = store.get_chunks(chunk_ids)
        mode = "hybrid-store" if store.vec_enabled else "fts-store"
        elapsed = (t1 - t0) * 1000.0
        if json_out:
            data = {
                "results": [{"id": rid, "score": sc} for rid, sc in results],
                "timings": {"total_ms": elapsed},
            }
            typer.echo(json.dumps(data))
            return
        typer.echo(_render_plain(q, k, mode, db, results, elapsed, texts=text_map))
        return
    except Exception:
        pass

    # Legacy fallback
    eng = _fallback_engine()
    hits, tm = eng.query(q, k=k, alpha=alpha, rerank=rerank)
    if json_out:
        data = {
            "results": [{"id": h.id, "score": h.score} for h in hits],
            "timings": tm.__dict__,
        }
        typer.echo(json.dumps(data))
        return
    results = [(h.id, h.score) for h in hits]
    text_map = {}
    if results:
        ids = [rid for rid, _ in results]
        qmarks = ",".join(["?"] * len(ids))
        rows = eng._conn.execute(  # type: ignore[attr-defined]
            f"SELECT id, text, meta FROM docs WHERE id IN ({qmarks})",
            ids,
        ).fetchall()
        for r in rows:
            rid = r["id"] if "id" in r.keys() else r[0]
            text = r["text"] if "text" in r.keys() else r[1]
            meta_raw = r["meta"] if "meta" in r.keys() else r[2]
            path = None
            try:
                if meta_raw:
                    meta = json.loads(meta_raw)
                    path = meta.get("path") if isinstance(meta, dict) else None
            except Exception:
                path = None
            text_map[rid] = {"text": text, "path": path}
    mode = "hybrid" if getattr(eng, "vec_enabled", False) else "fts"
    typer.echo(_render_plain(q, k, mode, db, results, tm.total_ms, texts=text_map))


@app.command()
def snapshot(to: str = typer.Argument(..., help="Absolute path to write snapshot")):
    try:
        c = _client()
        r = c.post("/snapshot", json={"to": to})
        r.raise_for_status()
        typer.echo(json.dumps(r.json()))
    except Exception:
        eng = _fallback_engine()
        res = eng.snapshot(to)
        typer.echo(json.dumps(res))


@app.command()
def info(json_out: bool = typer.Option(False, "--json", help="Output raw JSON")):
    """Print embedding backend and ONNX Runtime provider diagnostics."""
    import os as _os
    import onnxruntime as ort
    from .qq_embeddings import get_embedder

    data = {
        "env": {
            "QQ_EMBED_MODEL": _os.getenv("QQ_EMBED_MODEL"),
            "QQ_ORT_PROVIDER": _os.getenv("QQ_ORT_PROVIDER"),
            "QQ_ORT_PROVIDERS": _os.getenv("QQ_ORT_PROVIDERS"),
        },
        "ort": {
            "available_providers": list(ort.get_available_providers()),
            "version": getattr(ort, "__version__", None),
        },
        "embedding": {},
        "vector": {},
    }

    # Embedder info (may download model on first run)
    try:
        emb = get_embedder()
        _ = emb.encode(["warmup"])  # prime
        info = getattr(emb, "info", None)
        if info is not None:
            data["embedding"] = {
                "model": getattr(info, "model_name", None),
                "backend": getattr(info, "backend", None),
                "dim": getattr(info, "dim", None),
                "warmed": getattr(info, "warmed", None),
                "load_ms": getattr(info, "load_ms", None),
                "providers": getattr(info, "providers", None),
            }
    except Exception as e:  # pragma: no cover
        data["embedding"] = {"error": str(e)}

    # sqlite-vec availability (Python package)
    try:
        import sqlite_vec  # type: ignore

        data["vector"]["sqlite_vec"] = True
        data["vector"]["sqlite_vec_version"] = getattr(sqlite_vec, "__version__", None)
    except Exception:
        data["vector"]["sqlite_vec"] = False

    if json_out:
        typer.echo(json.dumps(data))
        return

    # Human-readable output
    lines = []
    lines.append("Embedding:")
    emb = data.get("embedding", {})
    if "error" in emb:
        lines.append(f"  error: {emb['error']}")
    else:
        lines.append(f"  model: {emb.get('model')}")
        lines.append(f"  backend: {emb.get('backend')}")
        lines.append(f"  dim: {emb.get('dim')}")
        lines.append(f"  providers: {', '.join(emb.get('providers') or []) if emb.get('providers') else 'n/a'}")
        lines.append(f"  load_ms: {int(emb.get('load_ms') or 0)}")
    lines.append("ONNX Runtime:")
    lines.append(f"  available_providers: {', '.join(data['ort'].get('available_providers') or [])}")
    lines.append(f"  version: {data['ort'].get('version')}")
    lines.append("Vector:")
    lines.append(f"  sqlite-vec: {data['vector'].get('sqlite_vec')}")
    if data['vector'].get('sqlite_vec_version'):
        lines.append(f"  sqlite-vec_version: {data['vector'].get('sqlite_vec_version')}")
    typer.echo("\n".join(lines))


# ---- index admin (new-arch store) ----

@index_app.command("stats")

def index_stats(json_out: bool = typer.Option(False, "--json", help="Output raw JSON")):
    try:
        from .store_sqlite import Store
    except Exception as e:
        typer.echo(json.dumps({"ok": False, "error": f"store import failed: {e}"}))
        raise typer.Exit(1)
    store = Store()
    st = store.stats()
    if json_out:
        typer.echo(json.dumps(st))
        return
    for k, v in st.items():
        typer.echo(f"{k}: {v}")


@index_app.command("export")

def index_export(out: str = typer.Argument(..., help="Output .sqlite path")):
    from .store_sqlite import Store

    store = Store()
    path = store.export_to(out)
    typer.echo(json.dumps({"ok": True, "out": path}))


@index_app.command("import")

def index_import(inp: str = typer.Argument(..., help="Input .sqlite path")):
    from .store_sqlite import Store

    store = Store()
    store.import_from(inp)
    typer.echo(json.dumps({"ok": True}))


app.add_typer(index_app, name="index")
