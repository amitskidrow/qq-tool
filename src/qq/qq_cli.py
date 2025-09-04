from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx
import typer

app = typer.Typer(add_completion=False, help="qq CLI (UDS client)")


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
    from .qq_embeddings import get_embedder

    db = os.getenv("QQ_DB", DEFAULT_DB_URI)
    emb = get_embedder()
    return Engine(db_uri=db, embedder=emb)


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
        # Fallback to in-process engine
        eng = _fallback_engine()
        for f in _iter_files(p):
            text = f.read_text(encoding="utf-8", errors="ignore")
            doc_id = _hash_id(f)
            eng.upsert(doc_id, text, {"path": str(f)})
            added += 1
        typer.echo(json.dumps({"ok": True, "ingested": added, "mode": "local"}))


def _render_plain(q: str, k: int, mode: str, db: str, results: List[Tuple[str, float]], elapsed_ms: float) -> str:
    from qq import __version__
    header = f"QQ {__version__} | mode={mode} | k={k} | db={db} | q=\"{q}\" | {int(elapsed_ms)}ms"
    lines = [header]
    for i, (rid, score) in enumerate(results, start=1):
        lines.append(f"{i} | score={score:.4f} | id={rid}")
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

    # Fallback to local engine
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
    mode = "hybrid" if getattr(eng, "vec_enabled", False) else "fts"
    typer.echo(_render_plain(q, k, mode, db, results, tm.total_ms))


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

