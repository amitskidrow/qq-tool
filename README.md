qq — local context server + CLI

Quickstart (Typesense-only)

- Start Typesense: `docker compose -f docker-typesense/docker-compose.yaml up -d`
- Install (uv tools): `uv tool install --from . qq` or with pipx: `pipx install .`
- First run: `qq setup` (bootstraps the `qq_docs` collection)
- Doctor: `qq doctor` (checks Typesense and model keys)
- Start API: `qq serve`

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

- Run the API as a background service so the embedding model stays in-memory.
  - Manual: `qq serve --host 127.0.0.1 --port 8787`
  - Systemd (user): see `scripts/systemd/qq.service` below.
- Use the CLI in remote mode so it calls the API instead of loading the model every run:
  - One-off: `qq query "..." --ns project:tradingapi --remote`
  - Always: set `QQ_REMOTE=1` and optionally `QQ_REMOTE_URL=http://127.0.0.1:8787`
- Choose a faster embedder if desired: `export QQ_EMBED_MODEL=all-MiniLM-L6-v2` (fast, 384-d).
- Optional GPU: `export QQ_EMBED_DEVICE=cuda` (falls back to CPU if unavailable).

Systemd (User) Service

1) Copy `scripts/systemd/qq.service` to `~/.config/systemd/user/qq.service` and adjust ExecStart and Environment as needed.
2) `systemctl --user daemon-reload`
3) `systemctl --user enable --now qq`
4) Check status: `systemctl --user status qq`

The Typesense container is configured with `restart: unless-stopped`, so it will come up automatically after reboot once set up via docker compose.
