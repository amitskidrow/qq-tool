# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/qq` (CLI in `qq_cli.py`, API in `qq_api.py`, engine in `qq_engine.py`).
- Config/state: `$HOME/.qq/` (overridable via `QQ_HOME`). Files include `config.yaml`, `sessions/`, `usage/`, `audit/`, `todos/`.
- Packaging: `pyproject.toml` (hatchling). Entry point script: `qq` → `qq.entry:main`.

## Build, Test, and Development Commands
- Install (editable): `pip install -e .` (Python ≥3.11). Alternative: `uv pip install -e .`.
- Install (tool): `uvx install --from . qq` or `pipx install .`.
- Quickstart: `docker compose up -d` (API only, via UDS); then `qq ingest path/to/docs/` and `qq query "..."`.
- Retrieval: `qq ingest path/to/docs/` then `qq query "your question"`.
- Release helper: `./deploy.sh` bumps patch in `pyproject.toml` and pushes.

## Coding Style & Naming Conventions
- Python style: 4‑space indent, 88–100 cols, type hints required for new public functions.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Imports: stdlib → third‑party → local. Prefer explicit exports in `__init__.py`.
- Lint/format: no enforced tool yet; prefer `ruff` and `black` locally before PRs.

## Testing Guidelines
- Framework: prefer `pytest`. Place tests under `tests/` mirroring `src/qq/` (e.g., `tests/test_cli.py`).
- Conventions: name files `test_*.py`; avoid network calls; the API runs locally over a UDS.
- Run (if configured): `pytest -q`; add coverage with `pytest --cov=qq`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits style (e.g., `feat: add hybrid merge`, `fix(cli): handle near-duplicates`). Release commits follow `chore(qq): release vX.Y.Z`.
- PRs: include concise description, rationale, and screenshots or sample CLI/API output when relevant. Link issues, list breaking changes, and add notes on config/env vars touched.

## Security & Configuration Tips
- No external services/keys required by default.
- Performance env vars: `QQ_EMBED_MODEL` (e.g., `sentence-transformers/all-MiniLM-L6-v2`), `QQ_UDS` (default `/run/qq.sock`).
- Do not commit credentials or `$HOME/.qq/` contents. Use `QQ_HOME` to point to a sandbox during tests.

## Docker Notes
- The API runs via a single container and binds a Unix domain socket at `/run/qq.sock` (bind mounted to host).
- Model/transformers cache is placed under `/dev/shm/hf` (tmpfs) to keep everything in RAM.
