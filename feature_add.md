Here’s a ready-to-use **instructions.md** you can paste into your repo for agentic CLIs (Claude Code, Copilot, Codex, etc.) to follow.

---

# instructions.md — Build a Minimal Python TUI to List & Inspect Ingested Data

## Purpose

Create a **small, well-architected, loosely-coupled** terminal UI that lets users **list, search, and preview raw ingested documents** without changing existing CLI/API behavior.

* UI framework: **Textual** (DataTable for rows, TextArea for preview, Input for search). Textual provides these widgets out of the box and runs on asyncio. ([Textual Documentation][1])
* Backend: **read-only HTTP API over Unix Domain Socket (UDS)** using `httpx` transport. Snapshot/offline mode uses direct **SQLite FTS5** queries. ([HTTPX][2])
* Streaming export: use FastAPI **StreamingResponse**; JSON responses can use **ORJSONResponse**. ([FastAPI][3])
* Optional vector debug: **sqlite-vec** for showing neighbors/ids from the vector index in snapshot mode. ([GitHub][4])

---

## Non-Goals

* No write/update/delete from the TUI.
* No TCP exposure; **UDS only** for the live service.
* No duplication of ranking logic in the client.

---

## High-Level Architecture (small & loosely coupled)

```
textual_tui/
  app.py                  # Textual App: layout, keybinds, paging
  backends/
    base.py               # Backend Protocol (list_docs, get_doc, export)
    live.py               # UDS HTTP client (httpx)
    snapshot.py           # SQLite FTS5 reader (read-only)
  models.py               # DocRow, DocDetail, Page (TypedDict/Pydantic)
  config.py               # QQ_SOCK (uds path), SNAPSHOT path
  tests/                  # Textual Pilot tests (headless)
```

**Separation of concerns**

* **UI** renders and manages interaction.
* **Backends** fetch data (live via UDS, or snapshot via SQLite).
* **Storage & ranking** remain server-side (or SQLite FTS5 in snapshot mode).

---

## Backend Contract (read-only)

Implement these API routes in your FastAPI service (or confirm they already exist):

* `GET /index/list?q=&like=&limit=&offset=`
  Returns: `{"rows":[{ "id","uri","title","size","updated","score","meta_summary"}]}`

  * Search uses **FTS5 `MATCH`** and ranks with **`bm25()`**. Keep pagination server-side. ([SQLite][5])

* `GET /index/get?id=…|uri=…`
  Returns: `{ "id","uri","title","meta","text" }`

* `GET /export?ids=…|q=…`
  Streams JSONL using **StreamingResponse**; prefer **ORJSONResponse** for regular JSON. ([FastAPI][6])

> Tip: Use `httpx` UDS transport on the client (`HTTPTransport(uds="…")`) so you don’t open any TCP ports. ([HTTPX][2])

---

## Snapshot Mode (offline / air-gapped)

When launched with `--snapshot /path/to.db`, the TUI bypasses HTTP and reads SQLite directly:

* If `q` is present:

  ```sql
  SELECT d.rowid AS id, d.uri, d.title, length(d.text) AS size, d.updated,
         bm25(docs_fts) AS score,
         substr(json_extract(d.meta,'$'),1,120) AS meta_summary
  FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid
  WHERE docs_fts MATCH ?
  ORDER BY score
  LIMIT ? OFFSET ?;
  ```

  Uses **FTS5** with **`MATCH`** + **`bm25()`** scoring. ([SQLite][5])

* Otherwise, list by recency:

  ```sql
  SELECT rowid AS id, uri, title, length(text) AS size, updated,
         0.0 AS score,
         substr(json_extract(meta,'$'),1,120) AS meta_summary
  FROM docs
  ORDER BY updated DESC
  LIMIT ? OFFSET ?;
  ```

* Optional: if **sqlite-vec** is loaded, you may JOIN on the `vec0` table to display vector ids/neighbors for debug tooling. ([GitHub][4])

---

## UX Requirements

* **Layout**:

  * Left (optional): facets/tree for URI prefixes.
  * Center: **DataTable** with columns `[id, uri, title, size, score]`.
  * Right: **TextArea** preview (raw text, soft wrap) + small JSON meta panel. ([Textual Documentation][1])

* **Keybinds**:

  * `/` focus search input; `Enter` applies server-side search.
  * `↑/↓/PgUp/PgDn` or `j/k` navigates rows; `Enter` opens selection.
  * `e` exports current result set (calls `/export`).
  * `b` toggles backend (live ↔ snapshot).

* **Behavior**:

  * Infinite scroll with `limit/offset` paging.
  * Don’t fetch full document bodies until selected.
  * Preview limits initial characters and offers “Load more” for very large texts.
  * Widgets operate independently on asyncio tasks (Textual). ([Textual Documentation][7])

---

## Implementation Steps (for the agent)

1. **Scaffold modules** as shown in the structure above (keep LOC small).
2. **Backend protocol** (`backends/base.py`) with:

   * `list_docs(q: str, like: str, limit: int, offset: int) -> List[DocRow]`
   * `get_doc(id: int | None, uri: str | None) -> DocDetail`
   * `export(...) -> Iterable[bytes]` (optional; hook to stream)
3. **Live backend** (`backends/live.py`):

   * Use `httpx.AsyncClient(transport=httpx.HTTPTransport(uds=QQ_SOCK))`.
   * Implement calls to `/index/list` and `/index/get`. ([HTTPX][2])
4. **Snapshot backend** (`backends/snapshot.py`):

   * Open SQLite read-only.
   * If `q`, run FTS5 `MATCH` + `bm25()`; else list by `updated DESC`. ([SQLite][5])
5. **App UI** (`app.py`):

   * Build a vertical layout: Input on top, then a horizontal split with DataTable and TextArea.
   * On mount: load first page. On search submit: reset offset and reload.
   * On row open: fetch full doc and fill TextArea.
   * Implement pagination trigger when near the end of DataTable.
6. **Export command** (optional):

   * Wire key `e` to call `/export` and stream JSONL to a file. Use **StreamingResponse** server-side. ([FastAPI][6])
7. **Testing**:

   * Use Textual’s test harness (**Pilot**) to drive headless UI flows (search → open → preview).
   * Add a snapshot test for the layout (stable columns, bindings). (Textual testing guidance is part of the framework docs/community.) ([Textual Documentation][8])

---

## Acceptance Criteria

* Launch: `qq tui` connects to UDS, or `qq tui --snapshot ./qq.db`.
* Listing: shows first page in DataTable; paging is smooth. ([Textual Documentation][1])
* Search: server-side FTS5 with ranked results (`bm25()`), consistent with CLI behavior. ([SQLite][5])
* Detail: selecting a row populates TextArea with the **raw** document text (soft wrapping). ([Textual Documentation][9])
* Export: streams JSONL via `/export`. ([FastAPI][6])
* No writes; no TCP; code footprint remains compact with clear boundaries.

---

## Configuration

* **Env**: `QQ_SOCK` (default `/tmp/qq.sock`), `QQ_SNAPSHOT` (optional path).
* **Flags**: `--snapshot /path/to.db` to force snapshot backend.
* **Packaging**: ship as `qq[ui]` extra; runs on standard Python (no native deps beyond SQLite).

---

## Reference Notes (for the agent)

* **Textual** widgets and architecture: **DataTable**, **TextArea**, widget/async model, gallery. ([Textual Documentation][1])
* **httpx** UDS transport: use `HTTPTransport(uds="…")` with `AsyncClient`. ([HTTPX][2])
* **FastAPI** responses: **StreamingResponse** for streaming, **ORJSONResponse** for fast JSON. ([FastAPI][3])
* **SQLite FTS5**: `MATCH` queries with **`bm25()`** ranking. ([SQLite][5])
* **sqlite-vec**: vector tables (`vec0`) for optional neighbor/ID debug. ([GitHub][4])

---

### Done = ✅

A slim Textual TUI that lists, searches, and previews ingested data over UDS (or reads a snapshot), with streaming export and zero behavior regressions.

[1]: https://textual.textualize.io/widgets/data_table/?utm_source=chatgpt.com "DataTable - Textual"
[2]: https://www.python-httpx.org/advanced/transports/?utm_source=chatgpt.com "Transports"
[3]: https://fastapi.tiangolo.com/advanced/custom-response/?utm_source=chatgpt.com "Custom Response - HTML, Stream, File, others - FastAPI"
[4]: https://github.com/asg017/sqlite-vec?utm_source=chatgpt.com "asg017/sqlite-vec: A vector search ..."
[5]: https://www.sqlite.org/fts5.html?utm_source=chatgpt.com "SQLite FTS5 Extension"
[6]: https://fastapi.tiangolo.com/reference/responses/?utm_source=chatgpt.com "Custom Response Classes - File, HTML, Redirect, ..."
[7]: https://textual.textualize.io/guide/widgets/?utm_source=chatgpt.com "Widgets - Textual"
[8]: https://textual.textualize.io/widget_gallery/?utm_source=chatgpt.com "Widgets - Textual"
[9]: https://textual.textualize.io/widgets/text_area/?utm_source=chatgpt.com "TextArea - Textual"
