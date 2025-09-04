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
import httpx

from . import __version__
from .api import build_app
from .config import CONFIG_PATH, ensure_dirs, load_config, save_config
from .errors import QQErrorCodes
from .store_typesense import TypesenseStore
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
from .compression import compress_texts
from .tokens import rough_token_count
from .todos import load_todos, add_todo, resolve_todo
# Typesense store handles CRUD directly


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


# --- int64 helpers for simhash storage ---
def _to_i64(n: int) -> int:
    """Convert unsigned 64-bit to signed 64-bit range."""
    n64 = n & ((1 << 64) - 1)
    return n64 - (1 << 64) if n64 >= (1 << 63) else n64


def _to_u64(n: int) -> int:
    """Normalize Python int to unsigned 64-bit representation for bit ops."""
    return n & ((1 << 64) - 1)


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


def _get_store() -> TypesenseStore:
    return TypesenseStore()


def _ts_ready() -> bool:
    try:
        store = _get_store()
        from .embeddings import embedding_dim
        store.ensure_schema(embedding_dim())
        return True
    except Exception:
        return False


    


    
@app.command()
def setup(ctx: typer.Context):
    """First-run checks; writes $HOME/.qq/config.yaml and verifies external services."""
    ensure_dirs()
    cfg = load_config()

    # Persist config if missing
    if not CONFIG_PATH.exists():
        save_config(cfg)

    # Prepare Typesense collection
    try:
        store = _get_store()
        from .embeddings import embedding_dim
        store.ensure_schema(embedding_dim())
        _echo_if_interactive(_interactive_from_ctx(ctx), "Typesense collection ready")
    except Exception as e:
        console.print(
            f"[yellow]{QQErrorCodes.ERR_VECTOR_UNAVAILABLE}[/yellow]: Typesense not ready ({e})",
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

    # Typesense
    try:
        store = _get_store()
        rep = store.doctor()
        if not rep.get("ok"):
            console.print(f"[red]{QQErrorCodes.ERR_VECTOR_UNAVAILABLE}[/red]: Typesense issues")
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
        doc_sim = _to_u64(simhash_text64(text))

        # find existing entries for this source (by uri)
        try:
            found = store.list_by_uri(str(fpath))
        except Exception as e:
            console.print(f"[red]Lookup error[/red]: {e}")
            raise typer.Exit(code=1)

        decision = "insert"
        if found:
            # If exact text match exists for any chunk, consider duplicate
            existing_text = "\n".join([str(r.get("text") or "") for r in found])
            existing_sim = _to_u64(simhash_text64(existing_text)) if existing_text else 0
            if content_hash(existing_text) == doc_hash:
                msg = f"{QQErrorCodes.ERR_DUPLICATE_DOC}: {fpath}"
                if interactive:
                    console.print(f"[yellow]{msg}[/yellow] -> skip")
                    decision = "skip"
                else:
                    console.print(f"[yellow]{msg}[/yellow]")
                    continue
            else:
                ham = hamming64(existing_sim, doc_sim) if existing_sim else 64
                if ham <= 3 and not allow_replace:
                    if interactive:
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
                store.delete_ids([str(r.get("id")) for r in found if r.get("id")])
            except Exception:
                pass

        # chunk and embed
        chunks = simple_chunks(text)
        vecs = embed_texts(chunks)

        # upsert
        try:
            n_chunks = 0
            for i, (ch, v) in enumerate(zip(chunks, vecs)):
                store.upsert_chunk(
                    uri=str(fpath),
                    namespace=ns_final,
                    text=ch,
                    vector=v,
                    title=fpath.name,
                    hash_=doc_hash,
                    simhash=_to_i64(doc_sim),
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
    hybrid: bool = typer.Option(False, "--hybrid", help="Combine keyword + dense via Typesense"),
    alpha: float = typer.Option(0.55, "--alpha", help="Hybrid fusion alpha (0..1)"),
    flat: int = typer.Option(20, "--flat", help="Exact scan cutoff (docs <= cutoff)"),
    compress: Optional[int] = typer.Option(None, "--compress", help="Percent to keep (1-99)"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Absolute token budget"),
    remote: bool = typer.Option(False, "--remote", help="Use running API server for low latency"),
    remote_url: Optional[str] = typer.Option(None, "--remote-url", help="Override API URL (default http://127.0.0.1:8787)"),
):
    """Vector mode: return top-N chunks with citations & metadata."""
    # Remote fast-path: always try the API first (single request), then fall back
    base = remote_url or os.getenv("QQ_REMOTE_URL") or "http://127.0.0.1:8787"
    if remote or True:
        try:
            ns_final = ns or infer_namespace()
            with httpx.Client(timeout=5.0) as client:
                r = client.get(
                    f"{base}/query",
                    params={"q": q, "ns": ns_final, "topk": topk, "hybrid": hybrid},
                )
                r.raise_for_status()
                data = r.json()
                out = data.get("results", [])
                payload = {"mode": "vector", "query": q, "ns": data.get("ns"), "topk": topk, "results": out}
                if compress is not None or max_tokens is not None:
                    texts = [str(i.get("snippet") or "") for i in out]
                    compressed_texts, ratio_applied = compress_texts(
                        texts,
                        ratio=(max(1, min(99, int(compress))) / 100.0) if compress is not None else None,
                        max_tokens=max_tokens,
                    )
                    for i, t in zip(out, compressed_texts):
                        i["snippet"] = t[:240]
                    payload["compressed"] = True
                    payload["compression_ratio"] = ratio_applied
                typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
                return
        except Exception as e:
            if remote:
                console.print(f"[yellow]Remote query failed[/yellow]: {e}. Falling back to local.")

    store = _get_store()

    # Embed query
    try:
        q_vec = embed_texts([q])[0]
    except Exception as e:
        console.print(f"[red]Embedding failed[/red]: {e}")
        raise typer.Exit(code=1)

    # Filter by ns if provided/inferred
    ns_final = ns or infer_namespace()

    # tune exact scan cutoff
    try:
        store.flat_search_cutoff = int(max(0, flat))
        if hybrid:
            results = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk, alpha=alpha)
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

    texts = [r.text or "" for r in results[:topk]]
    compressed = False
    ratio_applied = 1.0
    if compress is not None or max_tokens is not None:
        compressed = True
        if compress is not None:
            ratio = max(1, min(99, int(compress))) / 100.0
            texts, ratio_applied = compress_texts(texts, ratio=ratio)
        else:
            texts, ratio_applied = compress_texts(texts, max_tokens=max_tokens)

    for (r, t) in zip(results[:topk], texts):
        out.append({
            "id": r.id,
            "score": float(r.score or 0.0),
            "source": r.uri,
            "namespace": r.namespace,
            "snippet": (t or "")[:240],
        })
    payload = {"mode": "vector", "query": q, "ns": ns_final, "topk": topk, "results": out}
    if compressed:
        payload["compressed"] = True
        payload["compression_ratio"] = ratio_applied
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


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
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
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
            "uri": r.get("source"),
            "namespace": r.get("namespace"),
            "text": r.get("text"),
        }
        typer.echo(json.dumps({"id": r.get("id"), "payload": payload}, ensure_ascii=False))


@app.command()
def reindex():
    """Placeholder: would rebuild embeddings/indexes if needed."""
    typer.echo("ok")


@app.command()
def bench(
    q: str = typer.Argument("test query", help="Query to use for benchmarking"),
    ns: Optional[str] = typer.Option(None, "--ns"),
    iters: int = typer.Option(50, "--iters"),
    warm: int = typer.Option(5, "--warm"),
    topk: int = typer.Option(5, "--topk"),
    hybrid: bool = typer.Option(False, "--hybrid"),
    alpha: float = typer.Option(0.55, "--alpha"),
):
    """Benchmark end-to-end query latency (embed + search)."""
    import statistics
    import time

    store = _get_store()
    ns_final = ns or infer_namespace()

    # Warm embedder
    embed_texts([q])

    # Warm a few times
    for _ in range(max(0, warm)):
        q_vec = embed_texts([q])[0]
        if hybrid:
            _ = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk, alpha=alpha)
        else:
            _ = store.search_dense(q_vec, namespace=ns_final, topk=topk)

    # Timed runs
    times_ms: List[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        q_vec = embed_texts([q])[0]
        if hybrid:
            _ = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk, alpha=alpha)
        else:
            _ = store.search_dense(q_vec, namespace=ns_final, topk=topk)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    p50 = statistics.median(times_ms)
    p95 = statistics.quantiles(times_ms, n=20)[18] if len(times_ms) >= 20 else max(times_ms)
    payload = {"iters": iters, "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "min_ms": round(min(times_ms), 2), "max_ms": round(max(times_ms), 2)}
    typer.echo(json.dumps(payload))


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
