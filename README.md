qq — local context server + CLI

Quickstart

- Install (uvx): `uvx install --from . qq` or with pipx: `pipx install .`
- First run: `qq setup`
- Doctor: `qq doctor`
- Start API: `qq serve`

Notes

- Assumes Qdrant is running locally at `http://localhost:6333` (often via Docker), no auth.
- Default model provider is OpenAI with `gpt-5-mini`. Model names are not mapped; supported: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gemini-2.5-pro`.
- Daily token caps are enforced at a global (account) level and reset at midnight IST.
