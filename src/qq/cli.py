from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console

from . import __version__
from .api import build_app
from .config import CONFIG_PATH, ensure_dirs, load_config, save_config
from .errors import QQErrorCodes
from .store_sqlite import SqliteVecStore
from .util import infer_namespace, is_supported_file, read_text_file, now_ist
from .hashing import content_hash
from .simhash import simhash_text64, hamming64
from .chunking import simple_chunks
from .embeddings import embed_texts, embedding_dim
from .audit import append_audit
# Hybrid utils kept for potential rerank/analysis; not used directly here
from .sessions import create_session, load_session, save_session, list_sessions, close_session
from .usage import add_usage, get_usage, remaining
from .orchestrator import answer as orchestrated_answer
from .tokens import rough_token_count
from .todos import load_todos, add_todo, resolve_todo
# Qdrant-based CRUD removed; SQLite store handles CRUD directly


console = Console(stderr=True)
app = typer.Typer(add_completion=False, no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})


def _interactive_from_ctx(ctx: typer.Context) -> bool:
    try:
        obj = getattr(ctx, "obj", None)
        if isinstance(obj, dict):
            return bool(obj.get("interactive", False))
    except Exception:
        pass
    return False


def _echo_if_interactive(interactive: bool, msg: str) -> None:
    if interactive:
        console.print(msg)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-V", help="Show version and exit", is_eager=True),
    interactive: bool = typer.Option(False, "-i", help="Interactive mode for humans"),
):
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    # store interactive flag in context object
    if ctx.obj is None:
        ctx.obj = {}
    if isinstance(ctx.obj, dict):
        ctx.obj["interactive"] = interactive


@app.command("version")
def _version_cmd():
    """Print version and exit."""
    typer.echo(__version__)


def _get_store() -> SqliteVecStore:
    cfg = load_config()
    db_path = cfg.get("database", {}).get("path")
    if not db_path:
        raise RuntimeError("database.path not configured. Run 'qq config reset' to regenerate config.")
    return SqliteVecStore(Path(db_path))


def _db_ready() -> bool:
    try:
        store = _get_store()
        # Ensure schema exists with current embedding dimension if missing
        dim = store.get_dim()
        if dim is None:
            from .embeddings import embedding_dim
            store.ensure_schema(embedding_dim())
        else:
            # verify vec0 loads
            store.load_vec0()
        return True
    except Exception:
        return False


@app.command()
def db_init():
    """Initialize the SQLite database and store embedding dimension."""
    try:
        store = _get_store()
        from .embeddings import embedding_dim
        store.ensure_schema(embedding_dim())
        typer.echo(json.dumps({"ok": True, "path": str(store.db_path), "dim": store.get_dim()}))
    except Exception as e:
        console.print(f"[red]db.init failed[/red]: {e}")
        raise typer.Exit(code=1)


@app.command()
def rebuild_fts():
    """Rebuild the FTS index from docs."""
    try:
        store = _get_store()
        n = store.rebuild_fts()
        typer.echo(json.dumps({"rebuilt": n}))
    except Exception as e:
        console.print(f"[red]rebuild-fts failed[/red]: {e}")
        raise typer.Exit(code=1)


@app.command("migrate.qdrant")
def migrate_qdrant(
    url: str = typer.Option("http://localhost:6333", "--url"),
    collection: str = typer.Option("qq_data", "--collection"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    ns: Optional[str] = typer.Option(None, "--ns", help="Namespace filter; default: infer current"),
    batch: int = typer.Option(256, "--batch"),
):
    """One-time migration from Qdrant to SQLite. Requires 'qdrant-client' installed."""
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue  # type: ignore
    except Exception:
        console.print("[red]qdrant-client not installed[/red]: pip install qdrant-client to use this command")
        raise typer.Exit(code=2)

    store = _get_store()
    try:
        from .embeddings import embedding_dim
        store.ensure_schema(embedding_dim())
    except Exception:
        pass

    client = QdrantClient(url=url, api_key=api_key)
    ns_final = ns or infer_namespace()
    filt = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=ns_final))])
    next_offset = None
    migrated = 0
    with console.status("Migrating from Qdrant..."):
        while True:
            pts, next_offset = client.scroll(collection_name=collection, scroll_filter=filt, limit=batch, offset=next_offset, with_payload=True, with_vectors=True)
            if not pts:
                break
            for p in pts:
                pl = p.payload or {}
                text = pl.get("text") or ""
                uri = pl.get("source") or "migrate:qdrant"
                title = (pl.get("title") or str(uri).split("/")[-1]) if isinstance(uri, str) else None
                meta = {k: v for k, v in (pl or {}).items() if k not in {"text", "title"}}
                vec = np.array(p.vector, dtype=np.float32) if getattr(p, "vector", None) is not None else None
                if vec is None:
                    # fallback: embed if missing
                    vec = embed_texts([text])[0]
                store.upsert_chunk(
                    uri=str(uri),
                    namespace=pl.get("namespace") or ns_final,
                    text=text,
                    vector=vec,
                    title=title,
                    meta=meta,
                    hash_=pl.get("hash") or pl.get("doc_hash"),
                    simhash=int(pl.get("simhash") or 0) if pl.get("simhash") is not None else None,
                    created_at=None,
                    updated_at=None,
                )
                migrated += 1
            if next_offset is None:
                break
    typer.echo(json.dumps({"migrated": migrated, "namespace": ns_final}))
@app.command()
def setup(ctx: typer.Context):
    """First-run checks; writes $HOME/.qq/config.yaml and verifies external services."""
    ensure_dirs()
    cfg = load_config()

    # Persist config if missing
    if not CONFIG_PATH.exists():
        save_config(cfg)

    # Prepare SQLite store
    try:
        store = _get_store()
        from .embeddings import embedding_dim
        store.ensure_schema(embedding_dim())
        _echo_if_interactive(_interactive_from_ctx(ctx), f"SQLite store ready at {store.db_path}")
    except Exception as e:
        console.print(
            f"[yellow]{QQErrorCodes.ERR_VECTOR_UNAVAILABLE}[/yellow]: SQLite not ready ({e})",
            highlight=False,
        )

    # Keys presence (do not fail )
    openai_present = bool(os.getenv("OPENAI_API_KEY"))
    google_present = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if not openai_present:
        console.print(f"[yellow]{QQErrorCodes.ERR_OPENAI_KEY_MISSING}[/yellow]: OPENAI_API_KEY not set", highlight=False)
    if not google_present:
        console.print(f"[yellow]{QQErrorCodes.ERR_GOOGLE_KEY_MISSING}[/yellow]: GOOGLE_API_KEY/GEMINI_API_KEY not set", highlight=False)

    typer.echo(str(CONFIG_PATH))


@app.command()
def doctor():
    """Re-run checks; output actionable info. Non-zero exit on hard failures."""
    cfg = load_config()
    hard_fail = False

    # DB
    try:
        store = _get_store()
        rep = store.doctor()
        if not rep.get("ok"):
            console.print(f"[red]{QQErrorCodes.ERR_VECTOR_UNAVAILABLE}[/red]: SQLite issues")
            for c in rep.get("checks", []):
                if not c.get("ok"):
                    console.print(f" - {c.get('name')}: {c.get('detail')}")
            hard_fail = True
    except Exception as e:
        console.print(f"[red]{QQErrorCodes.ERR_VECTOR_UNAVAILABLE}[/red]: {e}")
        hard_fail = True

    # Model keys (warn only)
    if not os.getenv("OPENAI_API_KEY"):
        console.print(f"[yellow]{QQErrorCodes.ERR_OPENAI_KEY_MISSING}[/yellow]")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        console.print(f"[yellow]{QQErrorCodes.ERR_GOOGLE_KEY_MISSING}[/yellow]")

    raise typer.Exit(code=1 if hard_fail else 0)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8787, reload: bool = False):
    """Start API server."""
    uvicorn.run(build_app(), host=host, port=port, reload=reload, log_level="info")


@app.command()
def session(
    action: str = typer.Argument("list", help="list|show|close"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session id"),
):
    """Manage sessions."""
    if action == "list":
        typer.echo(json.dumps({"sessions": list_sessions()}))
        return
    if action == "show":
        if not session_id:
            console.print("Provide --session")
            raise typer.Exit(code=2)
        s = load_session(session_id)
        typer.echo(s.to_json())
        return
    if action == "close":
        if not session_id:
            console.print("Provide --session")
            raise typer.Exit(code=2)
        close_session(session_id)
        typer.echo(json.dumps({"closed": session_id}))
        return
    console.print("Unknown action. Use list|show|close")


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Prompt to ask (positional alternative to --ask)"),
    new: bool = typer.Option(False, "--new"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session id"),
    ask: Optional[str] = typer.Option(None, "--ask", help="Prompt to ask in session"),
    model: Optional[str] = typer.Option(None, "--model"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    ns: Optional[str] = typer.Option(None, "--ns"),
):
    """Create/resume chat sessions. When --ask provided, runs answer mode with retrieval and usage caps."""
    store = _get_store()

    # Allow positional message as a shorthand for --ask
    if ask is None and message is not None:
        ask = message

    if new:
        if not session_id:
            console.print("--session required with --new")
            raise typer.Exit(code=2)
        s = create_session(session_id, provider=provider, model=model)
        typer.echo(s.to_json())
        return

    if not session_id:
        console.print("Provide --session to resume (or create with --new --session <id>)")
        raise typer.Exit(code=2)
    s = load_session(session_id)
    # TTL check
    try:
        from .sessions import is_expired
        if is_expired(s):
            console.print(f"[red]{QQErrorCodes.ERR_SESSION_EXPIRED}[/red]: create a new session with --new")
            raise typer.Exit(code=3)
    except Exception:
        # if TTL parsing fails, proceed to avoid hard break
        pass
    if provider:
        s.provider = provider
    if model:
        s.model = model
    save_session(s)

    if not ask:
        typer.echo(json.dumps({"session": s.id, "provider": s.provider, "model": s.model}))
        return

    # Enforce daily caps, autoswitch if needed
    active_provider, active_model = s.provider, s.model
    rem = remaining(active_model)
    if rem is not None and rem <= 0 and active_provider == "openai":
        # autoswitch to gemini-2.5-pro
        s.autoswitches.append({"at": now_ist().isoformat(), "from": active_model, "to": "gemini-2.5-pro", "reason": "cap_exceeded"})
        active_provider, active_model = "google", "gemini-2.5-pro"
        save_session(s)

    # Orchestrate answer with retrieval
    try:
        out = orchestrated_answer(store, active_provider, active_model, ask, ns or infer_namespace())
    except Exception as e:
        console.print(f"[red]Answer failed[/red]: {e}")
        raise typer.Exit(code=1)

    # Rough token accounting
    input_tokens = rough_token_count(ask)
    output_tokens = rough_token_count(json.dumps(out))
    ok, _ = add_usage(active_provider, active_model, input_tokens + output_tokens)
    if not ok and active_provider == "openai":
        console.print(f"[yellow]{QQErrorCodes.ERR_TOKEN_CAP_EXCEEDED}[/yellow]: autoswitch next call")

    s.token_usage["input"] += input_tokens
    s.token_usage["output"] += output_tokens
    s.token_usage["total"] += input_tokens + output_tokens
    s.last_ns = ns or infer_namespace()
    save_session(s)

    # If the model suggests followups, add TODOs (optional behavior)
    followups = out.get("followups") or []
    created_todos = []
    for q in followups:
        tid = add_todo(q, session_id=s.id, ns=s.last_ns)
        created_todos.append(tid)

    payload = {"session": s.id, "provider": active_provider, "model": active_model, "answer": out, "todos": created_todos}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def ingest(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File or directory to ingest"),
    ns: Optional[str] = typer.Option(None, "--ns", help="Namespace override"),
    allow_replace: bool = typer.Option(False, "--allow-replace", help="Replace existing doc for same source"),
):
    """Add/refresh docs.

    Non-interactive: duplicate/near-duplicate raises error unless --allow-replace.
    Interactive (-i): prompts to skip/replace.
    """
    store = _get_store()

    # Discover files
    p = Path(path)
    files: list[Path] = []
    if p.is_file() and is_supported_file(p):
        files = [p]
    elif p.is_dir():
        for rp in p.rglob("*"):
            if is_supported_file(rp):
                files.append(rp)
    else:
        console.print(f"Path not found or unsupported: {path}")
        raise typer.Exit(code=2)

    # Ensure DB schema
    try:
        vec_dim = embedding_dim()
        store.ensure_schema(vec_dim)
    except Exception as e:
        console.print(f"[red]Failed to prepare database[/red]: {e}")
        raise typer.Exit(code=1)

    interactive = _interactive_from_ctx(ctx)
    ns_final = ns or infer_namespace()

    ingested = 0
    for fpath in files:
        text = read_text_file(fpath)
        doc_hash = content_hash(text)
        doc_sim = simhash_text64(text)

        # find existing entries for this source (by uri)
        try:
            found = store.list_by_uri(str(fpath))
        except Exception as e:
            console.print(f"[red]Lookup error[/red]: {e}")
            raise typer.Exit(code=1)

        decision = "insert"
        if found:
            # check exact dup using stored columns (SQLite rows)
            row0 = found[0]
            existing_hash = row0["hash"] if "hash" in row0.keys() else None
            try:
                existing_sim = int(row0["simhash"]) if row0["simhash"] is not None else 0
            except Exception:
                existing_sim = 0
            if existing_hash == doc_hash:
                msg = f"{QQErrorCodes.ERR_DUPLICATE_DOC}: {fpath}"
                if interactive:
                    console.print(f"[yellow]{msg}[/yellow] -> skip")
                    decision = "skip"
                else:
                    console.print(f"[yellow]{msg}[/yellow]")
                    continue
            else:
                # near-dup?
                ham = hamming64(existing_sim, doc_sim) if existing_sim else 64
                if ham <= 3 and not allow_replace:
                    if interactive:
                        # prompt
                        choice = typer.prompt(f"Near-duplicate for {fpath}. replace [y/N]?", default="n")
                        if choice.lower().startswith("y"):
                            decision = "replace"
                        else:
                            decision = "skip"
                    else:
                        console.print(f"[yellow]{QQErrorCodes.ERR_NEARDUP_DOC}[/yellow]: {fpath} (ham={ham}) use --allow-replace to override")
                        continue
                elif allow_replace:
                    decision = "replace"

        if decision == "skip":
            append_audit({
                "ts": now_ist().isoformat(),
                "action": "skip",
                "source": str(fpath),
                "ns": ns_final,
                "reason": "duplicate-or-user-skip",
            })
            continue

        # delete old entries if replacing
        if decision == "replace" and found:
            try:
                store.delete_ids([int(r["id"]) for r in found])
            except Exception:
                pass

        # chunk and embed
        chunks = simple_chunks(text)
        vecs = embed_texts(chunks)

        # upsert
        try:
            n_chunks = 0
            meta_base = {
                "type": "general_info",
                "importance": 1,
                "severity": 0,
                "priority": 0,
                "freshness_ts": int(now_ist().timestamp()),
                "doc_hash": doc_hash,
            }
            for i, (ch, v) in enumerate(zip(chunks, vecs)):
                meta = {**meta_base, "chunk_idx": i}
                store.upsert_chunk(
                    uri=str(fpath),
                    namespace=ns_final,
                    text=ch,
                    vector=v,
                    title=fpath.name,
                    meta=meta,
                    hash_=doc_hash,
                    simhash=int(doc_sim),
                    created_at=now_ist().isoformat(),
                    updated_at=now_ist().isoformat(),
                )
                n_chunks += 1
            ingested += 1
            append_audit({
                "ts": now_ist().isoformat(),
                "action": decision,
                "source": str(fpath),
                "ns": ns_final,
                "chunks": n_chunks,
                "doc_hash": doc_hash,
            })
            typer.echo(f"ingested {fpath} ({n_chunks} chunks)")
        except Exception as e:
            console.print(f"[red]Upsert failed[/red]: {e}")
            raise typer.Exit(code=1)

    typer.echo(f"done: {ingested} files")


@app.command()
def query(
    q: str = typer.Argument(..., help="Question or keywords"),
    ns: Optional[str] = typer.Option(None, "--ns"),
    topk: int = typer.Option(5, "--topk"),
    rerank: bool = typer.Option(False, "--rerank", help="Enable cross-encoder reranking of results"),
    hybrid: bool = typer.Option(False, "--hybrid", help="Combine dense vector and BM25 scores"),
    hybrid_pool: int = typer.Option(200, "--hybrid-pool", help="Max docs to consider for BM25 within namespace"),
):
    """Vector mode: return top-N chunks with citations & metadata."""
    store = _get_store()

    # Embed query
    try:
        q_vec = embed_texts([q])[0]
    except Exception as e:
        console.print(f"[red]Embedding failed[/red]: {e}")
        raise typer.Exit(code=1)

    # Filter by ns if provided/inferred
    ns_final = ns or infer_namespace()

    try:
        if hybrid:
            results = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk, pool=hybrid_pool)
        else:
            results = store.search_dense(q_vec, namespace=ns_final, topk=topk)
    except Exception as e:
        console.print(f"[red]Search failed[/red]: {e}")
        raise typer.Exit(code=1)

    out = []
    # Optional reranking (kept as-is); apply on candidate texts
    if rerank:
        try:
            from .rerank import rerank_pairs
            cfg_rk = load_config().get("retrieval", {}).get("reranker", {})
            model_name = cfg_rk.get("model", "BAAI/bge-reranker-large")
            cand_texts = [r.text or "" for r in results]
            order = rerank_pairs(model_name, q, cand_texts)
            results = [results[i] for i, _ in order][:topk]
        except Exception as e:
            console.print(f"[yellow]Rerank failed[/yellow]: {e}")
            results = results[:topk]

    for r in results[:topk]:
        out.append({
            "id": r.id,
            "score": float(r.score or 0.0),
            "source": r.uri,
            "namespace": r.namespace,
            "snippet": (r.text or "")[:240],
        })
    typer.echo(json.dumps({"mode": "vector", "query": q, "ns": ns_final, "topk": topk, "results": out}, ensure_ascii=False, indent=2))


@app.command()
def todo(action: str = typer.Argument("list", help="list|resolve"), tid: Optional[str] = None, answer: Optional[str] = None):
    if action == "list":
        typer.echo(json.dumps(load_todos(), ensure_ascii=False, indent=2))
        return
    if action == "resolve":
        if not tid or answer is None:
            console.print("Provide --tid and --answer")
            raise typer.Exit(code=2)
        ok = resolve_todo(tid, answer)
        typer.echo(json.dumps({"ok": ok, "id": tid}))
        return
    console.print("Unknown action. Use list|resolve")


@app.command()
def crud(
    ctx: typer.Context,
    delete: bool = typer.Option(False, "--delete"),
    ids: Optional[str] = typer.Option(None, "--ids", help="Comma-separated point ids for delete"),
    search: Optional[str] = typer.Option(None, "--search", help="Search query"),
    insert_text: Optional[str] = typer.Option(None, "--insert-text", help="Insert a text record"),
    ns: Optional[str] = typer.Option(None, "--ns"),
    type_: Optional[str] = typer.Option("general_info", "--type", help="Payload type for inserts"),
    importance: int = typer.Option(1, "--importance"),
    severity: int = typer.Option(0, "--severity"),
    force: bool = typer.Option(False, "--force", help="Override policy guard where applicable"),
):
    store = _get_store()
    ns_final = ns or infer_namespace()

    if delete:
        if not ids:
            console.print("--ids required for --delete")
            raise typer.Exit(code=2)
        try:
            id_list = [int(i.strip()) for i in ids.split(",") if i.strip()]
        except ValueError:
            console.print("ids must be integers for SQLite store")
            raise typer.Exit(code=2)
        deleted = store.delete_ids(id_list)
        typer.echo(json.dumps({"deleted": deleted}))
        return

    if search:
        try:
            q_vec = embed_texts([search])[0]
            res = store.search_hybrid(q_vec, search, namespace=ns_final, topk=10)
            payload = [
                {"id": r.id, "score": r.score, "text": r.text, "source": r.uri}
                for r in res
            ]
        except Exception as e:
            console.print(f"[red]Search failed[/red]: {e}")
            raise typer.Exit(code=1)
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if insert_text:
        payload = {
            "namespace": ns_final,
            "type": type_,
            "importance": importance,
            "severity": severity,
            "text": insert_text,
            "source": "crud:insert",
        }
        from .policy import guard_write
        ok, err = guard_write(payload, _interactive_from_ctx(ctx), force)
        if not ok:
            console.print(err)
            raise typer.Exit(code=3)
        v = embed_texts([insert_text])[0]
        doc_id = store.upsert_chunk(
            uri="crud:insert",
            namespace=ns_final,
            text=insert_text,
            vector=v,
            title=None,
            meta={"type": type_, "importance": importance, "severity": severity},
            hash_=None,
            simhash=None,
            created_at=now_ist().isoformat(),
            updated_at=now_ist().isoformat(),
        )
        typer.echo(json.dumps({"inserted": doc_id}))
        return

    console.print("Specify one of: --delete, --search, --insert-text")


@app.command()
def export(ns: Optional[str] = typer.Option(None, "--ns")):
    """Export namespace as JSONL to stdout."""
    store = _get_store()
    ns_final = ns or infer_namespace()
    rows = store.list_namespace(ns_final, limit=100000)
    for r in rows:
        payload = {
            "uri": r["uri"],
            "namespace": r["namespace"],
            "title": r["title"],
            "meta": json.loads(r["meta"] or "{}"),
            "text": r["text"],
            "hash": r["hash"],
            "simhash": r["simhash"],
        }
        typer.echo(json.dumps({"id": int(r["id"]), "payload": payload}, ensure_ascii=False))


@app.command()
def reindex():
    """Placeholder: would rebuild embeddings/indexes if needed."""
    typer.echo("ok")


@app.command()
def config(action: str = typer.Argument("print", help="print|reset")):
    """Print or reset config."""
    if action == "print":
        cfg = load_config()
        typer.echo(json.dumps(cfg, indent=2, ensure_ascii=False))
    elif action == "reset":
        from .config import default_config
        save_config(default_config())
        typer.echo(str(CONFIG_PATH))
    else:
        console.print("Unknown action. Use print|reset")


@app.command()
def model(action: str = typer.Argument("show", help="show|set"), name: Optional[str] = None, provider: Optional[str] = None):
    """Show or set default model/provider in config."""
    cfg = load_config()
    allowed_models = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gemini-2.5-pro"}
    allowed_providers = {"openai", "google"}

    if action == "show":
        m = cfg.get("client_model", {}).get("model")
        p = cfg.get("client_model", {}).get("provider")
        typer.echo(json.dumps({"provider": p, "model": m}))
        return

    if action == "set":
        if name is None and provider is None:
            console.print("Provide --name and/or --provider")
            raise typer.Exit(code=2)
        if name is not None and name not in allowed_models:
            console.print(f"Invalid model: {name}. Choose from {sorted(allowed_models)}")
            raise typer.Exit(code=2)
        if provider is not None and provider not in allowed_providers:
            console.print(f"Invalid provider: {provider}. Choose from {sorted(allowed_providers)}")
            raise typer.Exit(code=2)
        cm = cfg.setdefault("client_model", {})
        if name is not None:
            cm["model"] = name
        if provider is not None:
            cm["provider"] = provider
        save_config(cfg)
        typer.echo(json.dumps({"provider": cm.get("provider"), "model": cm.get("model")}))
        return

    console.print("Unknown action. Use show|set")


if __name__ == "__main__":
    app()
