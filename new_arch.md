# PRD — qq CLI: Hybrid Retrieval + Extractive Answer (v1)

## 0. Summary

Build a **local, deterministic retrieval system** for \~75–100 pages of mixed content (general + project‑specific + workflows + suggestions + model feedback). Default behavior: **return a single grounded paragraph** (≤120 words) with source metadata. Core pipeline: **FTS5 (BM25) + small embeddings + cross‑encoder rerank**, optional **graph‑lite boost** for relational queries, and optional **tiny LLM** for compression **without adding facts**.

---

## 1. Goals

* **G1:** High precision\@1 for “give me the relevant bit” queries.
* **G2:** Deterministic fallback; answers must be extractive with provenance.
* **G3:** Local‑first on Arch/CachyOS; minimal deps; reproducible.
* **G4:** Support relational/workflow queries via **graph‑lite** (no heavy KG).
* **G5:** Provide clean **CLI and JSON contracts** for agentic CLIs.

### Non‑Goals

* Generative multi‑paragraph synthesis.
* Full Graph‑RAG with LLM‑based relation extraction.
* Multi‑tenancy / remote index services.

---

## 2. Users & Usage

* **Primary:** Developer/operators and agentic CLIs (non‑interactive default).
* **Secondary:** Human analysts verifying citations.

---

## 3. Requirements

### Functional

* Ingest directories/files into SQLite, chunking by content type.
* Query returns **one paragraph** (≤120 words) + provenance, or **abstains**.
* Support modes: `lexical`, `semantic`, `hybrid`, `hybrid+rerank`, `graph‑boost`.
* Optional `--compress` using a tiny local LLM with strict guardrails.
* Export/import index; incremental re‑ingestion.

### Non‑Functional

* p50 latency ≤ 300 ms for 100 pages (no LLM); ≤ 800 ms with rerank; ≤ 1.5 s with compress.
* Deterministic results when `--no-llm`.
* Index size ≤ 50 MB for 100 pages typical.

---

## 4. System Overview

```
Files → Ingestor → [chunks, fts, vec, nodes, edges] (SQLite)
                                         │
Query → lexical (FTS5) ─┐
                        ├─ Hybrid merge/score → Top‑50 → Cross‑encoder rerank → Top‑1 → Answer
Query → semantic (vec) ─┘                                           │
                                          (optional) graph‑lite boost│
                                                           (optional) LLM compress (temp=0)
```

---

## 5. Data Model (SQLite)

```sql
-- documents
CREATE TABLE IF NOT EXISTS docs(
  doc_id TEXT PRIMARY KEY, title TEXT, type TEXT, source TEXT,
  created_at TEXT, updated_at TEXT, tags TEXT
);

-- chunks (paragraph/function/workflow/feedback)
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id INTEGER PRIMARY KEY,
  doc_id TEXT NOT NULL,
  section_path TEXT,
  kind TEXT CHECK(kind IN ('prose','code','workflow','feedback')),
  content TEXT NOT NULL,
  token_count INT,
  created_at TEXT, updated_at TEXT
);

-- lexical index
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  content, section_path, doc_id UNINDEXED, chunk_id UNINDEXED,
  tokenize='unicode61'
);

-- vector index (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(
  id INTEGER PRIMARY KEY,        -- chunk_id
  content BLOB,                  -- float32[] (normalized)
  dims INTEGER
);

-- graph‑lite (optional)
CREATE TABLE IF NOT EXISTS nodes(
  node_id INTEGER PRIMARY KEY,
  kind TEXT,   -- function|class|concept|workflow|feedback
  name TEXT,
  ref_chunk_id INT,
  extra JSON
);
CREATE TABLE IF NOT EXISTS edges(
  src INT, dst INT, rel TEXT, weight REAL DEFAULT 1.0,
  PRIMARY KEY (src, dst, rel)
);
```

---

## 6. Ingestion

### Chunking

* **Prose:** 300–450 tokens, \~15% overlap; carry `section_path` from headings.
* **Code/API:** chunk per function/class; include signature + docstring with body intro.
* **Workflows:** step‑wise chunks; preserve ordering via `section_path` (e.g., `Deploy > Step 3`).
* **Model feedback:** store as `kind='feedback'` with `extra` (author, run\_id, labels, stance).

### Process

1. Normalize (Unicode, whitespace), strip boilerplate.
2. Chunk by type; compute `token_count`.
3. Insert into `chunks`; upsert `docs`.
4. Insert into `fts` from `chunks`.
5. Embed each chunk (small local model) → store normalized float32 in `vec`.
6. (Optional) Create `nodes`/`edges` from rule‑based extractors (no LLM):

   * symbols from code (regex), headers as `concept` nodes, workflow steps as nodes.

### Idempotency

* De‑dup via hash of normalized `content`; retain first; link dupes via `edges(rel='duplicate')`.
* Incremental updates: upsert by `(doc_id, section_path, hash)`.

---

## 7. Retrieval Pipeline

**Defaults:** `K_lex=30`, `K_sem=30`, `K_merge=50`, `WORD_LIMIT=120`.

1. **Lexical**: `fts MATCH :query` → top‑Kₗ (score `bm25`).
2. **Semantic**: embed `query` → ANN over `vec` → top‑Kₛ (score `cosine`).
3. **Hybrid merge**: union candidates; min‑max normalize `bm25` within candidate set.

   * Score: `S = 0.55 * cosine + 0.45 * bm25_norm`.
   * Keep top‑K (50).
4. **Graph‑lite boost (conditional)**:

   * If query contains relational cues (regex: `depends on|calls|before|after|owner|implements`).
   * Map terms to `nodes.name` via FTS; expand 1‑hop `edges` → chunk set `G`.
   * Boost: `S := S + 0.05` for chunks in `G` (cap at +0.1 total).
5. **Rerank** (cross‑encoder, CPU‑friendly MiniLM/BGE): score top‑20; pick top‑3.
6. **Select**: choose top‑1; if `S_top < 0.12` **and** CE score < `τ₂`, **abstain**.
7. **Answer**: return raw paragraph; optionally `--compress` via tiny LLM (temp=0). If LLM adds facts or exceeds `WORD_LIMIT`, **fallback** to raw.

---

## 8. CLI & JSON Contracts

### Ingest

```
qq ingest --path <dir|file> [--type prose|code|workflow|feedback] \
  [--graph-lite] [--reindex] [--tags k:v,k:v]
```

**Output (JSON):**

```json
{"docs": 12, "chunks": 874, "duplicates": 31, "embedded": 874, "nodes": 220, "edges": 340}
```

### Query

```
qq query --q "<question>" \
  --mode lexical|semantic|hybrid \
  [--rerank ce] [--graph-boost] [--max-words 120] [--compress] [--no-llm]
```

**Output (JSON):**

```json
{
  "answer": "<<=120 words, extractive or compressed>",
  "source": {"chunk_id": 123, "doc_id": "prj-spec.md", "section_path": "API > Auth"},
  "scores": {"hybrid": 0.72, "ce": 0.63},
  "policy": "extractive_only",
  "abstained": false
}
```

**Abstain example:** `{"answer":"No relevant paragraph found.", "abstained":true}`

### Admin

```
qq index stats   # prints counts, size, last updated
qq index export --out index.sqlite
qq index import --in index.sqlite
```

---

## 9. Optional Tiny LLM (Compression Only)

* Runtime: Ollama (3B/7B instruct; Q4 quant). Temperature **0**; `num_ctx ≥ 2048`.
* System: “You are a careful editor. Only compress and clarify; do not add or change facts.”
* User: includes query + snippet; output **≤ WORD\_LIMIT**; else fallback to raw.

---

## 10. Telemetry & Quality Gates

* Metrics: precision\@1, abstain rate, p50/p95 latency, index size, recall\@50 (debug), rerank hit rate, LLM‑fallback rate.
* Logging (structured): `{q_id, mode, top_ids, S_top, ce_top, abstain}`.
* Threshold tuning: adjust `0.55/0.45` and `0.12` via offline eval (see §11).

---

## 11. Evaluation Plan

* **Gold set:** 100 queries spanning facts, workflows, decisions, code, relational.
* **Pipelines:** (A) FTS5, (B) Hybrid, (C) Hybrid+CE, (D) (C)+Graph‑boost, (E) (C)+Compress.
* **Targets:**

  * precision\@1 ≥ 0.78 on gold set.
  * abstain rate 5–15% (prefer abstain over wrong).
  * p50 latency ≤ 800 ms with CE; ≤ 1.5 s with compress.
* **Acceptance:** ship simplest pipeline that meets targets (likely C or D).

---

## 12. Operations

* **Build:** Arch/CachyOS; deps via `uv`/`pipx`; SQLite with FTS5 + sqlite‑vec; CE model packaged.
* **Index lifecycle:** nightly `--reindex` optional; otherwise incremental.
* **Backup:** periodic `index.sqlite` snapshot; export/import supported.
* **Security:** local‑only; no external calls when `--no-llm`.

---

## 13. Risks & Mitigations

* **R1:** Vocabulary mismatch → add embeddings + hybrid, synonyms table.
* **R2:** Near‑duplicate chunks → de‑dup + rerank.
* **R3:** LLM drift → temp=0 + strict prompts + fallback.
* **R4:** Graph noise → rule‑based extractors only; small boosts.

---

## 14. Milestones

* **M1 (Day 1–2):** DB schema + ingestor (prose/code) + FTS5 search.
* **M2 (Day 3–4):** Embeddings + hybrid scoring + CLI.
* **M3 (Day 5):** CE reranker + thresholds + JSON contract.
* **M4 (Day 6):** Graph‑lite extractors + boost path.
* **M5 (Day 7):** Eval suite + tuning + docs.

---

## 15. Agentic CLI Task Map

1. `IngestTask`: normalize → chunk → insert → embed → graph‑lite (optional).
2. `QueryTask`: retrieve (lex/sem) → hybrid → graph‑boost (if any) → rerank → select → compress (optional) → emit JSON.
3. `EvalTask`: run gold set across pipelines; compute metrics; update thresholds.
4. `AdminTask`: export/import/stats; validate index integrity.

---

## 16. Defaults (tunable constants)

```
WORD_LIMIT = 120
K_lex = 30; K_sem = 30; K_merge = 50
HYBRID_WEIGHTS = (cosine=0.55, bm25=0.45)
GRAPH_BOOST = +0.05 (max +0.1)
S_THRESH = 0.12   # hybrid score abstention guard
CE_THRESH = 0.20  # reranker score guard
```

**End of PRD**
