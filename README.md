qq — local context server + CLI

Quickstart (Docker Compose)

- Start both services: `docker compose up -d`
- Install CLI (uv tools): `uv tool install --from . qq` or with pipx: `pipx install .`
- First run: `qq setup` (bootstraps the `qq_docs` collection)
- Doctor: `qq doctor` (checks Typesense and model keys)

Ingest & Query

- Ingest: `qq ingest path/to/docs --ns project:tradingapi`
- Query (dense): `qq query "..." --ns project:tradingapi --topk 5`
- Hybrid: `qq query "..." --ns project:tradingapi --topk 5 --hybrid --alpha 0.55`
- Compression: `qq query "..." --ns project:tradingapi --compress 50`

Notes

- Retrieval backend is Typesense (dev-only, local). The client embeds a fixed dev key (`tsdev`) and talks to `http://localhost:8108`.
- Collection `qq_docs` schema: id (string), namespace (facet), source (facet), text, embedding (float[], num_dim from embedder), created_at, updated_at.
- Default model provider is OpenAI with `gpt-5-mini`. Supported: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gemini-2.5-pro`.
- Daily token caps are enforced at a global (account) level and reset at midnight IST.

Low-Latency Mode (<1s queries)

- Keep the API hot via Docker Compose. The API runs alongside Typesense and keeps the embedding model in memory.
- The CLI auto-detects the API at `http://127.0.0.1:8787` and uses remote mode by default when it’s healthy.
  - Force on: `qq query "..." --remote` or `export QQ_REMOTE=1`
  - Override URL: `export QQ_REMOTE_URL=http://127.0.0.1:8787`
- Choose a faster embedder if desired: `export QQ_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2` (fast, 384-d).
- Optional GPU: `export QQ_EMBED_DEVICE=cuda` (falls back to CPU if unavailable).

Docker Compose Details

- Services: `typesense` on `127.0.0.1:8108`, `qq-api` on `127.0.0.1:8787`.
- Persistence: Typesense data under named volume `typesense-data`.
- Caches: model caches under `hf-cache` and `torch-cache` volumes to avoid re-downloads.
- Healthchecks: compose waits for Typesense health before starting the API.
- Tuning via env vars (set in compose or shell): `QQ_EMBED_MODEL`, `QQ_EMBED_DEVICE`, `QQ_TYPESENSE_*` (host/port/protocol/api key).

Notes

- The compose is local-only and binds ports to `127.0.0.1`.
- Dev-only Typesense key is `tsdev`. For a private machine, no extra auth is configured.
- The legacy `docker-typesense/` setup has been removed. Use the root `docker-compose.yml` for both services.

Build notes

- The API image preinstalls CPU-only PyTorch to avoid downloading large CUDA wheels during build and uses no-build-isolation to stabilize dependency resolution.
- If you have a CUDA GPU available, you can override at runtime with `QQ_EMBED_DEVICE=cuda` and switch to a GPU-enabled image/profile later.
