qq — local context server + CLI

Quickstart

- Install (uv tools): `uv tool install --from . qq` or with pipx: `pipx install .`
- First run: `qq setup`
- Doctor: `qq doctor`
- Start API: `qq serve`

Notes

- Uses a local SQLite database with the sqlite-vec extension for vector search and FTS5 for lexical search. No external DB is required.
- Default model provider is OpenAI with `gpt-5-mini`. Model names are not mapped; supported: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gemini-2.5-pro`.
- Daily token caps are enforced at a global (account) level and reset at midnight IST.
