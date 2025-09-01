# qq — Product Requirements Document (PRD)

> **Version:** 1.1
> **Owner:** amit / core tools
> **Status:** Draft for build (amended per latest directions)

---

## At-a-glance feature list (for quick review)

* **Zero-friction install (no Docker):** install via **pipx** or **uvx** from repo URL; pure-CLI bootstrap, local-only.
* **Silent-by-default CLI** for agents; **interactive mode** (`qq -i ...`) for humans.
* **Local vector store**: single Qdrant collection with payload filters; hybrid search (dense + sparse/BM25) and optional local reranker.
* **Namespaces**: `global` and `project:<name>` with indexed payload fields; automatic project detection and tagging.
* **Info Types**: `general_info`, `global_approach`, `project_workflow_thread`, `project_info`.
* **Importance & Severity** fields that affect ranking and policy handling.
* **Freshness bias** and **priority** tie-breaker in scoring.
* **Ingest preflight**: content hash + SimHash near-dup detection; diff → skip/replace/merge (interactive) or error (non-interactive).
* **Answer modes**:

  * **Vector mode**: return top-N chunks with citations & metadata.
  * **Answer mode** (default): synthesized guidance (Answer Contract JSON) with citations & optional command plan.
* **Built-in model clients (switchable):** `gpt-5` / `gpt-5-mini` (default) / `gpt-5-nano` **and** `gemini-2.5-pro`. Switchable via CLI & config.
* **Sessioned chat for CLIs** (24h lifetime): create/list/resume sessions; per-session system prompt; token accounting; IST timezone.
* **Model-driven CRUD tools** (tool/function calls): `kb.search/insert/update/delete` with write gates, SimHash checks, and audit log.
* **System prompt control**: configurable via CLI & config; prompt nudges agents to ask follow-ups and query qq for missing context.
* **Policy-lite (optional)**: OPA/Rego allow/deny/nudge using importance/severity & rules (e.g., prefer `uv`, avoid `pip`, prefer `we/kk`).
* **Token usage limits & autoswitch**: per-session caps (gpt-5=250k, mini/nano=2M). Auto-switch to `gemini-2.5-pro` when exceeded.
* **Interactive TODOs for unresolvable queries**: model creates human-review TODOs; guides agent to ask the user; TODOs are tracked & resolvable.
* **Robust error codes** for deterministic agent retries.
* **Maintenance**: export, reindex, doctor; append-only audit logs for writes.
* **Optional MCP adapter** for plug-and-play with IDE/agent shells.

---

## 1. Overview & Goals

**qq** is a small, local "context server + CLI" that centralizes team conventions and project knowledge for agentic CLIs and humans. It is:

* **CLI-first & local-first** (no Docker): install via pipx/uvx; config in `$HOME/.qq/`.
* **Agent-friendly** (silent, predictable errors), **human-friendly** (interactive wizard, sessions, TODOs).
* **Low-debt** (single collection, typed contracts, small modules).

Non-goals unchanged: not a hosted multi-tenant service; not a heavy KG/UI.

---

## 2. Personas

* **Agentic CLI** (primary): Claude Code, Gemini CLI, Qwen, etc. Uses chat sessions & non-interactive API.
* **Developer** (secondary): uses interactive chat & ingest wizards; resolves TODOs.
* **Maintainer** (secondary): curates knowledge, resolves conflicts, manages policies/prompts.

---

## 3. Principles

* **Simplicity over knobs** (defaults + config > flags).
* **Determinism** (clear error codes; no hidden prompts unless `-i`).
* **Trust** (citations, audit, explicit policy nudges).
* **Progressive enhancement** (reranker, OPA, MCP are optional).

---

## 4. Architecture

### 4.1 Components (no Docker)

```
+----------------------+       +---------------------------+
|      qq CLI          |<----->|      qq API (FastAPI)     |
| Typer; pipx/uvx      |       | /setup /ingest /query     |
| Silent by default    |       | /chat  /crud  /model      |
+----------+-----------+       | /prompt /session /health  |
           |                   +------------+--------------+
           v                                |
+-------------------+                      v
|  Config Manager   |            +---------------------------+
| $HOME/.qq/config  |            |  Model Orchestrator       |
+-------------------+            |  (OpenAI & Gemini clients)|
                                 |  - model switch (5/mini/  |
                                 |    nano / gemini-2.5-pro) |
                                 |  - tool calls (CRUD)      |
                                 |  - session mgmt & tokens  |
                                 +------------+--------------+
                                              |
                           tool CRUD + search v
                                 +---------------------------+
                                 |  Vector Layer (Qdrant)    |
                                 |  single collection        |
                                 |  hybrid + filters + CRUD  |
                                 +------------+--------------+
                                              |
                                              v
                                 +---------------------------+
                                 | Embeddings & Reranker     |
                                 | BGE-M3, bge-reranker      |
                                 +------------+--------------+
                                              |
                                              v
                                 +---------------------------+
                                 | Policy-lite (OPA)         |
                                 +---------------------------+
```

### 4.2 Key choices

* **Install** via pipx/uvx; no container runtime required.
* **Single collection + payload filters** for namespaces/projects/types.
* **Hybrid retrieval** (dense+sparse) + optional cross-encoder rerank.
* **Sessions** (24h, IST) with system prompts, token accounting, autoswitch.
* **Tool-enabled model** for CRUD with gates (allow-write, severity rules, SimHash) and **interactive TODO** creation when blocked.

---

## 5. CLI Spec (updated)

```
qq setup                       # first-run checks; writes $HOME/.qq/config.yaml
qq doctor                      # re-run checks; actionable fixes
qq serve                       # start API
qq ingest <path>               # add/refresh docs; non-interactive; auto-detect
qq query "<q>"                 # one-shot; mode=answer by default
qq chat --new --session <id>   # create a 24h session (IST); optional --model ...
qq chat --session <id> --ask "..."   # resume session and ask
qq session list|show|close <id>
qq model set <gpt-5|gpt-5-mini|gpt-5-nano|gemini-2.5-pro>
qq prompt show|set <file|text> # set system prompt (applies per-session unless overridden)
qq crud --delete --ids ...     # explicit admin CRUD (bypass model tools)
qq todo list|resolve <id>      # view/resolve interactive TODOs left by model
qq export [--ns …] > dump.jsonl
qq reindex
qq config [print|edit|reset]
qq -i <subcmd …>               # interactive wizard for humans
```

### 5.1 Flags (still rare)

* `--ns <global|project:NAME>`
* `--mode vector|answer`
* `--topk <N>`
* `--allow-write` (permit model CRUD in non-interactive)
* `--force` (override for high-severity/global edits in interactive)
* `--model <name>` and `--provider <openai|google>` (overrides for a call)

---

## 6. Sessions (IST, 24h)

* **Create**: `qq chat --new --session <id>` → initializes store in `$HOME/.qq/sessions/<id>.json` with `created_at` in IST, active model, system prompt, token budget counters.
* **Resume**: `qq chat --session <id> --ask "..."` → continues until 24h since `created_at` (IST) or explicit `close`.
* **List**: `qq session list` → shows open/expired; `show` prints metadata and token usage; `close` marks ended.
* **System prompt** applied per-session; can be overridden at creation or mid-session.
* **Token accounting & caps** per session:

  * `gpt-5`: **250k** tokens cap
  * `gpt-5-mini` / `gpt-5-nano`: **2,000,000** tokens cap
  * When cap exceeded, **autoswitch** to `gemini-2.5-pro` and record the switch in session metadata.

---

## 7. System Prompt (internal model)

* **Configurable** via `qq prompt set <file|text>` and config `client_model.system_prompt`.
* Prompt **encourages follow-ups**: instruct the agent to:

  1. Ask qq clarifying questions first;
  2. Prefer querying the vector DB using `kb.search` when context is missing;
  3. If blocked after reasonable attempts, **create a TODO** for the human and clearly instruct the agent to ask the user the specific question.

---

## 8. Model Clients & Switching

* **Providers**: `openai` (gpt-5 family) and `google` (`gemini-2.5-pro`).
* **Defaults**: provider=openai, model=`gpt-5-mini`.
* **Switching**: `qq model set ...` (persists to config) or `--model/--provider` per call.
* **Structured outputs** (Answer Contract) and **tool/function calls** (CRUD palette) supported for both providers.
* **Env keys**: `OPENAI_API_KEY` for OpenAI; `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Gemini. Config keys optional; env takes precedence.
* **Autoswitch on caps**: when token cap hit, switch to `gemini-2.5-pro` for the remainder of the session.

---

## 9. Data Model (unchanged + session stores)

**Vector collection** payload unchanged (namespace/type/importance/severity/etc.).
**Sessions store**:

```
{
  "id": "<session-id>",
  "created_at_ist": "YYYY-MM-DDTHH:mm:ss+05:30",
  "expires_at_ist": "<+24h>",
  "provider": "openai|google",
  "model": "gpt-5-mini|gpt-5|gpt-5-nano|gemini-2.5-pro",
  "system_prompt": "...",
  "token_usage": {"input": n, "output": n, "total": n},
  "autoswitches": [ {"at": ts, "from": modelA, "to": modelB, "reason": "cap_exceeded"} ],
  "last_ns": "project:<name>|global"
}
```

---

## 10. Ingest & Query (no change to basics)

* **Ingest**: normalize → content hash + SimHash → dup/near-dup handling → chunk → embed → upsert(id=hash). Non-interactive errors vs interactive diff.
* **Query**: embed → hybrid search → rerank → freshness/importance scoring → (optional) policy → Answer Contract JSON.

---

## 11. Model-driven CRUD (tools) & Guards

**Tools**: `kb.search`, `kb.insert`, `kb.update`, `kb.delete` with strict JSON schemas.
**Guards**:

* `--allow-write` required for non-interactive writes; otherwise `ERR_WRITE_NOT_ALLOWED`.
* OPA severity gate: edits to `global_approach` with `severity>=2` require `--force` in interactive or are denied in non-interactive (`ERR_POLICY_GUARD`).
* SimHash preflight blocks conflicting inserts/updates in non-interactive; interactive offers skip/replace/merge.
* **Audit**: all writes → `$HOME/.qq/audit/*.jsonl`.

---

## 12. Interactive TODOs (human handoff)

* When the model cannot resolve a user/agent query (missing policy, ambiguous instructions, unknown project facts), it **creates a TODO**:

  * `qq todo list` shows pending items with session, namespace, and the exact question to ask the user.
  * The model also **instructs the agent** to ask the user that question directly.
  * Developer resolves with `qq todo resolve <id> --answer "..."` which the model can then ingest (with appropriate type/importance/severity) or attach as a note.

---

## 13. Configuration (amendments)

Additions to `$HOME/.qq/config.yaml`:

```yaml
client_model:
  provider: openai           # or google
  model: gpt-5-mini          # default
  system_prompt: "..."       # optional default
  token_caps:
    gpt-5: 250000
    gpt-5-mini: 2000000
    gpt-5-nano: 2000000
  autoswitch: gemini-2.5-pro
sessions:
  timezone: Asia/Kolkata
  ttl_hours: 24
```

---

## 14. Error Codes (expanded)

* `ERR_QUERY_ARGS_MISSING` (namespace cannot be inferred)
* `ERR_DUPLICATE_DOC` (exact hash exists)
* `ERR_NEARDUP_DOC` (near-duplicate detected)
* `ERR_OPENAI_KEY_MISSING` / `ERR_GOOGLE_KEY_MISSING`
* `ERR_MODEL_UNAVAILABLE`
* `ERR_SESSION_EXPIRED`
* `ERR_TOKEN_CAP_EXCEEDED` (switch performed; call again)
* `ERR_WRITE_NOT_ALLOWED`
* `ERR_POLICY_GUARD`

---

## 15. Install & Setup (no Docker)

**Install (examples):**

```
# pipx
pipx install git+https://github.com/your-org/qq.git

# uvx
activate uv (if needed) then:
uvx install --from git+https://github.com/your-org/qq.git qq
```

**First run:**

```
qq setup   # writes $HOME/.qq/config.yaml; probes vector DB, models, and policy config
```

---

## 16. Observability & Performance (unchanged)

* Audit JSONL for writes; optional /metrics.
* p95 query ≤ 500 ms (excluding model synthesis); ingest ≥ 500 docs/min (local SSD, parallel embeds).

---

## 17. Rollout Plan (updated highlights)

* **Sprint 1**: pipx/uvx packaging; config & setup; vector store & ingest with SimHash; vector query.
* **Sprint 2**: hybrid retrieval + reranker; answer mode (OpenAI client, `gpt-5-mini` default) + Answer Contract.
* **Sprint 3**: sessions (IST, 24h) + token accounting + autoswitch; model switching incl. `gemini-2.5-pro`.
* **Sprint 4**: policy-lite (OPA); model-driven CRUD + guards + audit; interactive TODOs.
* **Sprint 5 (opt)**: MCP adapter; metrics.

---

## 18. Definition of Done (incremental)

* Installable via pipx/uvx; `qq setup` produces a valid config and passes checks.
* Sessions working (create/list/resume/close), IST timestamps, token caps & autoswitch.
* Model switching across `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gemini-2.5-pro` via CLI & config.
* Ingest preflight (hash + SimHash), single collection with payload filters & indexes.
* Hybrid retrieval + optional reranker; Answer Contract JSON with citations.
* CRUD tools callable by model, gated & audited; policy-lite annotations.
* Interactive TODOs end-to-end (create → list → resolve → ingest/update as needed).
