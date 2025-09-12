from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Iterable

import httpx
import typer

app = typer.Typer(add_completion=False, help="qq CLI (server-only, UDS client)")
index_app = typer.Typer(help="Index admin commands (server store)")


def _uds_path() -> str:
    return os.getenv("QQ_UDS", "/run/qq.sock")


def _client() -> httpx.Client:
    transport = httpx.HTTPTransport(uds=_uds_path())
    return httpx.Client(transport=transport, base_url="http://qq.local", timeout=30.0)


def _hash_id(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8"))
    return h.hexdigest()


def _iter_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
        return
    skip_dirs = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}
    for f in p.rglob("*"):
        parts = set(f.parts)
        if parts & skip_dirs:
            continue
        if f.is_file() and f.suffix.lower() in {".txt", ".md", ".rst", ".log"}:
            yield f


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
):
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
            payload = {"id": doc_id, "text": text, "meta": {"path": str(f), "title": f.name}}
            r = c.post("/upsert", json=payload)
            r.raise_for_status()
            added += 1
        typer.echo(json.dumps({"ok": True, "ingested": added, "mode": "server"}))
    except Exception as e:
        typer.echo(json.dumps({"ok": False, "error": f"server ingest failed: {e}"}))
        raise typer.Exit(1)


"""
Server-only CLI utilities.
"""


@app.command()
def query(
    q: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(5, "--k", help="Top-K results"),
    alpha: float = typer.Option(0.55, "--alpha", help="Dense weight [0,1] (hybrid only)"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranker"),
    graph_boost: bool = typer.Option(False, "--graph-boost/--no-graph-boost", help="Enable graph-lite boost"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    try:
        c = _client()
        r = c.post(
            "/query2",
            json={
                "q": q,
                "k": k,
                "alpha": alpha,
                "rerank": rerank,
                "graph_boost": graph_boost,
            },
        )
        r.raise_for_status()
        data = r.json()
        if json_out:
            typer.echo(json.dumps(data))
            return
        if data.get("abstained"):
            typer.echo("No relevant paragraph found.")
            return
        answer = data.get("answer") or ""
        scores = data.get("scores") or {}
        src = data.get("source") or {}
        lines: list[str] = []
        lines.append(f"answer (<=120w):\n{answer}")
        meta_bits = []
        if src.get("doc_id") is not None:
            meta_bits.append(f"doc_id={src.get('doc_id')}")
        if src.get("section_path"):
            meta_bits.append(f"section_path={src.get('section_path')}")
        if src.get("chunk_id") is not None:
            meta_bits.append(f"chunk_id={src.get('chunk_id')}")
        if scores.get("hybrid") is not None:
            meta_bits.append(f"hybrid={float(scores['hybrid']):.3f}")
        if scores.get("ce") is not None:
            meta_bits.append(f"ce={float(scores['ce']):.3f}")
        if meta_bits:
            lines.append(" | ".join(meta_bits))
        typer.echo("\n".join(lines))
    except Exception as e:
        typer.echo(json.dumps({"ok": False, "error": f"server query failed: {e}"}))
        raise typer.Exit(1)


@app.command()
def doctor(json_out: bool = typer.Option(False, "--json", help="Output raw JSON report")):
    """Check local server integration via UDS and basic ingest/query."""
    report: dict[str, object] = {"ok": False, "uds": {}, "api": {}}
    uds = _uds_path()
    uds_ok = False
    try:
        st = os.stat(uds)
        is_sock = stat.S_ISSOCK(st.st_mode)
        perm = oct(st.st_mode & 0o777)
        report["uds"] = {"path": uds, "exists": True, "is_socket": is_sock, "perm": perm}
        uds_ok = is_sock
    except FileNotFoundError:
        report["uds"] = {"path": uds, "exists": False}
        if json_out:
            typer.echo(json.dumps(report))
        else:
            typer.echo("UDS not found; is the server running?")
        raise typer.Exit(1)

    api = {"connect": False, "index_stats": None, "smoke": None}
    try:
        c = _client()
        # connectivity: index stats
        resp = c.get("/index/stats")
        api["connect"] = resp.status_code == 200
        api["index_stats"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
        # smoke ingest + query
        doc_id = f"qq:doctor:{int(time.time())}"
        text = "Doctor smoke test: Jupiter is the largest planet."
        up = c.post("/upsert", json={"id": doc_id, "text": text, "meta": {"path": "doctor"}})
        up.raise_for_status()
        qr = c.post("/query2", json={"q": "largest planet", "k": 3, "alpha": 0.55, "rerank": False, "graph_boost": False})
        qr.raise_for_status()
        data = qr.json()
        api["smoke"] = data
        report["api"] = api
        report["ok"] = uds_ok and api["connect"] and bool(data and not data.get("abstained"))
    except Exception as e:
        api["error"] = str(e)
        report["api"] = api
        report["ok"] = False
        if json_out:
            typer.echo(json.dumps(report))
            raise typer.Exit(1)
        else:
            typer.echo(f"doctor failed: {e}")
            raise typer.Exit(1)

    if json_out:
        typer.echo(json.dumps(report))
    else:
        typer.echo("doctor: OK" if report.get("ok") else "doctor: FAIL")


@app.command()
def snapshot(to: str = typer.Argument(..., help="Absolute path to write snapshot")):
    c = _client()
    r = c.post("/snapshot", json={"to": to})
    r.raise_for_status()
    typer.echo(json.dumps(r.json()))


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


# ---- index admin (server store) ----

@index_app.command("stats")

def index_stats(json_out: bool = typer.Option(False, "--json", help="Output raw JSON")):
    c = _client()
    r = c.get("/index/stats")
    if r.status_code != 200:
        typer.echo(json.dumps({"ok": False, "error": r.text}))
        raise typer.Exit(1)
    st = r.json()
    if json_out:
        typer.echo(json.dumps(st))
        return
    for k, v in st.items():
        typer.echo(f"{k}: {v}")


@index_app.command("export")

def index_export(out: str = typer.Argument(..., help="Output .sqlite path on server")):
    c = _client()
    r = c.post("/index/export", json={"out": out})
    if r.status_code != 200:
        typer.echo(json.dumps({"ok": False, "error": r.text}))
        raise typer.Exit(1)
    typer.echo(json.dumps(r.json()))


@index_app.command("import")

def index_import(inp: str = typer.Argument(..., help="Input .sqlite path on server")):
    c = _client()
    r = c.post("/index/import", json={"inp": inp})
    if r.status_code != 200:
        typer.echo(json.dumps({"ok": False, "error": r.text}))
        raise typer.Exit(1)
    typer.echo(json.dumps(r.json()))


app.add_typer(index_app, name="index")
