qq — local in‑memory retrieval (UDS API + CLI)

Overview

- Engine: SQLite (in‑memory) + sqlite‑vec (exact KNN) + FTS5 (BM25)
- Embeddings: all‑MiniLM‑L6‑v2 (384‑d) via ONNX Runtime; model warmed on startup
- API: FastAPI served only over a Unix domain socket (UDS) at `/run/qq.sock`
- CLI: talks to the API over the same UDS; no TCP ports; no external services
- Persistence: everything in RAM; explicit `snapshot` writes a `.db` file on demand

Quickstart

- Start API: `docker compose up -d`
- Install CLI: `uv tool install --from . qq` (or `pipx install .`)
- Ingest docs: `qq ingest path/to/docs/`
- Query: `qq query "your question"`
- Snapshot: `qq snapshot /absolute/path/to/qq.db`

Architecture

- In‑Memory DB: `file:qqmem?mode=memory&cache=shared` (shared across connections in the API process)
- Tables:
  - `docs(id TEXT PRIMARY KEY, text TEXT, meta TEXT)`
  - `docs_fts` (FTS5) with `id UNINDEXED, text`
  - `vec` (sqlite‑vec) with `embedding float[384]` and a `vec_map(id<->rowid)` mapping
- Fusion: vector similarity and BM25 blended with `alpha=0.55`, default `k=6`
- Reranker: off by default to prioritize latency

Docker Compose

- Single service `qq-api` running uvicorn with `--uds /run/qq.sock`
- `/run` bind‑mounted from host so CLI can connect
- `HF_HOME` and model cache under `/dev/shm/hf` (tmpfs) to keep model/cache in RAM

Performance (SLO)

- Target: p95 < 1s for `qq query` on a 10–15 page corpus (warm server)
- The API logs per‑request timings: `embed_ms`, `vec_ms`, `fts_ms`, `fuse_ms`, `rerank_ms`, `total_ms`

Notes

- No Typesense/Qdrant or external vector DBs
- No TCP ports opened; API reachable only via `/run/qq.sock`
- Set `QQ_UDS` to override the UDS path (default `/run/qq.sock`)
