# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/qq` (CLI in `cli.py`, API in `api.py`, retrieval in `store_typesense.py`, models in `models/`).
- Config/state: `$HOME/.qq/` (overridable via `QQ_HOME`). Files include `config.yaml`, `sessions/`, `usage/`, `audit/`, `todos/`.
- Packaging: `pyproject.toml` (hatchling). Entry point script: `qq` → `qq.entry:main`.

## Build, Test, and Development Commands
- Install (editable): `pip install -e .` (Python ≥3.10). Alternative: `uv pip install -e .`.
- Install (tool): `uvx install --from . qq` or `pipx install .`.
- Quickstart: `docker compose up -d` (Typesense + API), `qq setup` (writes config, ensures Typesense collection and checks keys), `qq doctor` (re-checks).
- Low-latency: API runs via compose and stays hot; CLI auto-detects API and uses remote mode when healthy. Force with `qq query --remote` or `QQ_REMOTE=1`.
- Retrieval: `qq ingest path/to/docs/` then `qq query "your question"`.
- Release helper: `./deploy.sh` bumps patch in `pyproject.toml` and pushes.

## Coding Style & Naming Conventions
- Python style: 4‑space indent, 88–100 cols, type hints required for new public functions.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Imports: stdlib → third‑party → local. Prefer explicit exports in `__init__.py`.
- Lint/format: no enforced tool yet; prefer `ruff` and `black` locally before PRs.

## Testing Guidelines
- Framework: prefer `pytest`. Place tests under `tests/` mirroring `src/qq/` (e.g., `tests/test_cli.py`).
- Conventions: name files `test_*.py`; use fixtures; avoid network calls (mock LLM clients). Retrieval uses a local Typesense instance; for tests, prefer mocking the store or spinning up Typesense via the provided docker-compose.
- Run (if configured): `pytest -q`; add coverage with `pytest --cov=qq`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits style (e.g., `feat: add hybrid merge`, `fix(cli): handle near-duplicates`). Release commits follow `chore(qq): release vX.Y.Z`.
- PRs: include concise description, rationale, and screenshots or sample CLI/API output when relevant. Link issues, list breaking changes, and add notes on config/env vars touched.

## Security & Configuration Tips
- Required services/keys: `OPENAI_API_KEY` and/or `GOOGLE_API_KEY`/`GEMINI_API_KEY` for models.
- Performance env vars: `QQ_EMBED_MODEL` (e.g., `sentence-transformers/all-MiniLM-L6-v2`), `QQ_EMBED_DEVICE` (`cpu` | `cuda`), `QQ_REMOTE`, `QQ_REMOTE_URL`, and Typesense connection overrides `QQ_TYPESENSE_HOST|PORT|PROTOCOL|API_KEY`.
- Do not commit credentials or `$HOME/.qq/` contents. Use `QQ_HOME` to point to a sandbox during tests.

## Docker Notes
- The legacy `docker-typesense/` compose has been removed; use the root `docker-compose.yml` to run Typesense and the qq API together.
- The API Dockerfile preinstalls CPU-only PyTorch and disables build isolation in pip to stabilize dependency resolution and avoid large CUDA wheel downloads during `docker compose build`.
