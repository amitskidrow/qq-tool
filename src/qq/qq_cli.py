from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import typer

app = typer.Typer(add_completion=False, help="qq CLI (UDS client)")


def _uds_path() -> str:
    return os.getenv("QQ_UDS", "/run/qq.sock")


def _client() -> httpx.Client:
    transport = httpx.HTTPTransport(uds=_uds_path())
    # Base URL host/path is ignored with UDS, but required by httpx
    return httpx.Client(transport=transport, base_url="http://qq.local", timeout=30.0)


def _hash_id(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8"))
    return h.hexdigest()


def _iter_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
    else:
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".rst", ".log"}:
                yield f


@app.command()
def ingest(path: str = typer.Argument(..., help="File or directory to ingest")):
    p = Path(path)
    if not p.exists():
        typer.echo(json.dumps({"ok": False, "error": f"path not found: {path}"}))
        raise typer.Exit(2)
    c = _client()
    added = 0
    for f in _iter_files(p):
        text = f.read_text(encoding="utf-8", errors="ignore")
        doc_id = _hash_id(f)
        payload = {"id": doc_id, "text": text, "meta": {"path": str(f)}}
        r = c.post("/upsert", json=payload)
        r.raise_for_status()
        added += 1
    typer.echo(json.dumps({"ok": True, "ingested": added}))


@app.command()
def query(
    q: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(6, "--k", help="Top-K results"),
    alpha: float = typer.Option(0.55, "--alpha", help="Dense weight [0,1]"),
    rerank: bool = typer.Option(False, "--rerank/--no-rerank", help="Enable reranker"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty table output"),
):
    c = _client()
    r = c.post("/query", json={"q": q, "k": k, "alpha": alpha, "rerank": rerank})
    r.raise_for_status()
    data = r.json()
    if pretty:
        try:
            from rich.table import Table
            from rich.console import Console

            tbl = Table(title=f"qq results (k={k}, alpha={alpha})")
            tbl.add_column("rank")
            tbl.add_column("id")
            tbl.add_column("score")
            for i, hit in enumerate(data.get("results", []), start=1):
                tbl.add_row(str(i), str(hit.get("id")), f"{hit.get('score', 0):.4f}")
            Console().print(tbl)
        except Exception:
            typer.echo(json.dumps(data))
    else:
        typer.echo(json.dumps(data))


@app.command()
def snapshot(to: str = typer.Argument(..., help="Absolute path to write snapshot")):
    c = _client()
    r = c.post("/snapshot", json={"to": to})
    r.raise_for_status()
    typer.echo(json.dumps(r.json()))


