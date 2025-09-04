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

AMD/CPU Compatibility

- Default runtime is CPU-only via ONNX Runtime and works on AMD CPUs and GPUs without NVIDIA/CUDA.
- To try AMD GPU acceleration, install `onnxruntime-rocm` in your environment and set a provider:
  - `pip install onnxruntime-rocm`
  - `export QQ_ORT_PROVIDER=ROCMExecutionProvider` (falls back to CPU if unavailable)
  - Verify with: `python -c 'import onnxruntime as o; print(o.get_available_providers())'`
- Docker image provided here is CPU-only; ROCm in containers requires a ROCm base image and exposing `/dev/kfd` and `/dev/dri`.

ROCm Compose Sample

- A sample ROCm overlay is provided: `docker-compose.rocm.yml` with `Dockerfile.rocm`.
- Start it (if your host has ROCm and the devices are available):
  - `docker compose -f docker-compose.rocm.yml up -d`
- The service binds `/run/qq.sock` and requests devices `/dev/kfd` and `/dev/dri`.
- Environment sets `QQ_ORT_PROVIDER=ROCMExecutionProvider`.

CLI Diagnostics

- Show active embedding backend and ONNX Runtime providers:
  - `qq info` (human-readable) or `qq info --json` (machine-readable)
- Useful envs: `QQ_EMBED_MODEL`, `QQ_ORT_PROVIDER`, `QQ_ORT_PROVIDERS`.
