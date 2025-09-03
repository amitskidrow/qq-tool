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
