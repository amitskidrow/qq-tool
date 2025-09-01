# Goal

Replace Qdrant with a **zero‑ops, embedded** store using **SQLite + sqlite‑vec + FTS5** while preserving or improving retrieval quality for a tiny corpus (≈10–15 pages → 20–80 chunks). The output here is a *step‑by‑step plan* tuned for automation by agentic CLIs (Claude Code, Codex, Gemini CLI, etc.).

---

## Snapshot / Constraints

* Target OS: **Arch Linux (CachyOS)**; Python 3.11+; uv/pipx preferred.
* Corpus size: **micro** (exact KNN viable; perfect recall expected).
* Quality priority: **Good embeddings + hybrid BM25 + optional reranker**.
* No Docker; **single file DB** at `~/.qq/qq.db`; WAL enabled.
* Backward compatible CLI surface for qq (ingest/query/export).

---

## Architecture (High Level)

```
+------------------+       +-----------------------------+
|  Ingestion CLI   | ----> |  Embedder (Sentence-Tfmrs)  |
+------------------+       +-----------------------------+
          |                                |
          v                                v
+-----------------+      +-------------------------------+
|  SQLite: docs   | <--> |  FTS5 (BM25) + sqlite-vec     |
|  (uri, title,   |      |  vec_index(embedding float[N])|
|   meta, text)   |      +-------------------------------+
+-----------------+                  ^
          |                          |
          v                          |
   Reranker (opt)  <-----------------+
          |
          v
      Results
```

---

## Data Model (SQLite)

**Files**: single DB at `~/.qq/qq.db`

```sql
-- base table
CREATE TABLE IF NOT EXISTS docs(
  id     INTEGER PRIMARY KEY,
  uri    TEXT UNIQUE,   -- path/url/key
  title  TEXT,
  meta   TEXT,          -- JSON
  text   TEXT           -- full text
);

-- FTS5: external content, BM25 ranking
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
USING fts5(text, content='docs', content_rowid='id');

-- sqlite-vec: one row per doc id; dim decided at init
-- example for 384-d model:
CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
USING vec0(embedding float[384]);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_uri ON docs(uri);
PRAGMA journal_mode=WAL;     -- durability + concurrency
PRAGMA synchronous=NORMAL;   -- balance speed/safety
```

> **Dim strategy:** at first boot, detect selected embedding model’s output dimension and create `vec_index` accordingly (e.g., 384 for `all-MiniLM-L6-v2`; 768 for `bge-small-en-v1.5`; 1024 if you keep current). Store the chosen `dim` into a tiny `settings` table.

```sql
CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
-- store dim: INSERT OR REPLACE INTO settings(k,v) VALUES('dim','384');
```

---

## Core Retrieval Algorithm (Hybrid + Optional Rerank)

1. **Vector pass (exact)**: cosine similarity over all chunks via `sqlite-vec`.
2. **Lexical pass**: FTS5 `bm25()` over the same corpus.
3. **Score fusion**: `score = α * sim + (1-α) * bm25_norm` with `α ∈ [0,1]` (default 0.5).
4. **Optional rerank**: top‑K (e.g., 20) re‑ordered by a cross‑encoder (local model) for maximum quality; keep K=5–8.

Recommended defaults:

* `K_vector = 20`, `K_lex = 20`, `K_final = 6`, `α = 0.55`.
* Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, small) or `bge-reranker-base` if GPU is available.

---

## CLI/Module Surface (Agent‑Friendly)

**Subcommands (stable names):**

```bash
qq db.init                 # create ~/.qq/qq.db, detect dim, create tables
qq ingest <path|uri>...    # add/update docs; auto-chunk; embed; upsert
qq rebuild-fts             # rebuild FTS external content after bulk ops
qq query "<text>"           # hybrid query; prints JSONL results
qq export.sqlite <out.db>  # copy or vacuum/backup the DB
qq migrate.qdrant <url>    # ONE-TIME: pull from existing Qdrant and insert
qq doctor                  # integrity checks + vacuum + pragma report
```

**Environment / Config** (YAML @ `~/.qq/config.yml`):

```yaml
vector_store:
  driver: sqlite
  db_path: ~/.qq/qq.db
  alpha: 0.55
  top_k: 6
embedding:
  model: all-MiniLM-L6-v2   # or your existing model
  normalize: true
chunking:
  size: 800
  overlap: 120
reranker:
  enabled: false            # true to enable
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

**JSONL output** from `qq query` (easy for agents to parse):

```json
{"rank":1,"score":0.812,"uri":"doc://intro","title":"Intro","snippet":"...","meta":{"tag":"x"}}
{"rank":2,"score":0.788,"uri":"doc://api","title":"API","snippet":"..."}
```

---

## Implementation Tasks (ordered for agents)

### T0 — Bootstrap Store Adapter

* [ ] Create module `src/qq/store_sqlite.py` with a `SqliteVecStore` class exposing:

  * `init(dim:int, model_name:str)` (or infer `dim` by probing embedder once)
  * `upsert(items: Iterable[Dict])`
  * `search(query:str, k:int, alpha:float) -> List[Dict]`
  * `rebuild_fts()`; `backup(to_path)`; `doctor()`
* [ ] On first run: create tables, set PRAGMAs, persist `settings.dim`.

### T1 — Embedder & Chunker

* [ ] `src/qq/embedding.py`: wrapper around SentenceTransformers with `normalize_embeddings=True`.
* [ ] `src/qq/chunker.py`: token/char window chunking (size=800, overlap=120) + metadata (uri, title, page, section).

### T2 — Ingestion Pipeline

* [ ] `qq ingest` accepts files/dirs/URIs; detects type (md, txt, pdf via text extractor) → chunks → `store.upsert()`.
* [ ] After bulk ingest, call `rebuild_fts` once.

### T3 — Query Pipeline (Hybrid)

* [ ] Embed query; vector KNN via sqlite‑vec (LIMIT 2K); lexical BM25 (LIMIT 2K); min‑max normalize BM25; fuse by α.
* [ ] If reranker enabled: form `(query, doc_text)` pairs and rerank top 20.
* [ ] Emit JSONL and a pretty table (for humans) using Rich.

### T4 — Qdrant Migration (Optional, one‑time)

* [ ] `qq migrate.qdrant http://localhost:6333/collections/qq_data`:

  * page through points; read payload `{uri,title,meta,text}` + vectors
  * insert into `docs` and `vec_index` (respecting `rowid=id` mapping)
  * `rebuild_fts`
* [ ] Remove Qdrant docker files from repo after success.

### T5 — Tests / CI

* [ ] Unit: chunker, embedder dim check, sqlite‑vec CRUD, FTS rebuild, hybrid fusion math.
* [ ] Golden e2e: seed 10 docs → query 10 prompts → verify top‑3 contain expected URIs.
* [ ] Bench: measure cold vs warm query times; assert p95 < 25 ms on laptop CPU.

### T6 — Observability & Maintenance

* [ ] Structured logging (JSON) for `ingest`, `query`, `migrate`: timings for `embed`, `knn`, `fts`, `rerank`.
* [ ] `qq doctor` prints PRAGMA settings, counts, orphan checks, and runs `VACUUM` if flagged `--fix`.

---

## Code Templates (copy‑ready)

### Python: store bootstrap

```python
# src/qq/store_sqlite.py
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple
import numpy as np

class SqliteVecStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

    def load_vec0(self):
        import sqlite_vec
        sqlite_vec.load(self.conn)

    def exec(self, sql: str, args: Tuple = ()):  # small helper
        return self.conn.execute(sql, args)

    def commit(self):
        self.conn.commit()
```

### Python: dimension detection (run once at db.init)

```python
# src/qq/db_init.py
from sentence_transformers import SentenceTransformer
from qq.store_sqlite import SqliteVecStore

def db_init(db_path, model_name):
    st = SentenceTransformer(model_name)
    dim = st.get_sentence_embedding_dimension()

    s = SqliteVecStore(db_path)
    s.load_vec0()

    s.exec("""
    CREATE TABLE IF NOT EXISTS docs(
      id INTEGER PRIMARY KEY, uri TEXT UNIQUE, title TEXT, meta TEXT, text TEXT);
    """)
    s.exec("""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
    USING fts5(text, content='docs', content_rowid='id');
    """)
    s.exec(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
    USING vec0(embedding float[{dim}]);
    """)
    s.exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_uri ON docs(uri)")
    s.exec("CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT)")
    s.exec("INSERT OR REPLACE INTO settings(k,v) VALUES('dim', ?)", (str(dim),))
    s.commit()
    return dim
```

### Python: hybrid query (vector + BM25 + fusion)

```python
# src/qq/search.py
import json, sqlite3, numpy as np
from typing import List, Dict

def hybrid_search(conn: sqlite3.Connection, query_vec, query_text: str, k=6, alpha=0.55):
    # vector pass
    vec_rows = conn.execute(
        """
        SELECT d.id, 1.0 - v.distance AS sim
        FROM vec_index v JOIN docs d ON d.id=v.rowid
        WHERE embedding MATCH ? ORDER BY v.distance LIMIT ?
        """, (json.dumps(query_vec.tolist()), max(k*2, 20))
    ).fetchall()

    # lexical pass
    bm_rows = conn.execute(
        """
        SELECT d.id, bm25(docs_fts) AS s
        FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
        WHERE docs_fts MATCH ? ORDER BY s LIMIT ?
        """, (query_text, max(k*2, 20))
    ).fetchall()

    bm = {i: s for i, s in bm_rows}
    if bm:
        scores = np.array(list(bm.values()), dtype=np.float32)
        lo, hi = float(scores.min()), float(scores.max())
        rng = (hi - lo) or 1.0
        for i in bm: bm[i] = 1.0 - ((bm[i] - lo)/rng)

    sim = {i: s for i, s in vec_rows}
    cand = set(sim) | set(bm)
    fused = [(i, alpha*sim.get(i,0.0) + (1-alpha)*bm.get(i,0.0)) for i in cand]
    fused.sort(key=lambda t: t[1], reverse=True)
    top_ids = [i for i,_ in fused[:k]]

    if not top_ids: return []
    qmarks = ",".join(["?"]*len(top_ids))
    rows = conn.execute(f"SELECT id,uri,title,meta,text FROM docs WHERE id IN ({qmarks})", top_ids).fetchall()
    out: List[Dict] = []
    for i,uri,title,meta,text in rows:
        out.append({"id":i,"uri":uri,"title":title,"meta":json.loads(meta or '{}'),"text":text})
    return out
```

> **Note:** Keep embeddings normalized at encode time. Then `1.0 - distance` from sqlite‑vec approximates cosine similarity.

---

## Packaging / Install (Arch‑friendly)

Prefer **uv** or **pipx**.

```bash
# using uv
uv venv && source .venv/bin/activate
uv pip install -e .

# or pipx (system-wide isolated)
pipx install .

# ensure sqlite-vec native ext loads (packaged wheel). If building from source:
pacman -S base-devel sqlite
# (sqlite-vec provides wheels for manylinux; source build rarely needed.)
```

---

## Makefile Targets (for agents)

```make
init:
	python -m qq.db_init --model all-MiniLM-L6-v2

ingest:
	python -m qq.cli ingest $(PATHS)

query:
	python -m qq.cli query "$(Q)" --k 6 --alpha 0.55

rebuild-fts:
	python -m qq.cli rebuild-fts

migrate-qdrant:
	python -m qq.cli migrate.qdrant $(QDRANT_URL)

doctor:
	python -m qq.cli doctor --fix
```

---

## Evaluation & Quality Gates

* **Gold set**: 10 queries mapped to expected top‑3 URIs.
* **Metrics**: Hit\@1 ≥ 0.6, Hit\@3 ≥ 0.9 on gold set; p95 latency < 25 ms CPU‑only.
* **A/B**: Run old Qdrant pipeline vs new hybrid; assert equal or better Hits.

Command for agents:

```bash
qq eval.run --gold ./eval/gold.jsonl --n 10 --report ./eval/report.json
```

---

## Observability & Maintenance

* Structured logs (`--json` flag) with timings: `embed_ms`, `knn_ms`, `fts_ms`, `rerank_ms`.
* `qq doctor` checks: table presence, `settings.dim` consistency, orphan rows (`docs` vs `vec_index`), FTS integrity, WAL checkpoint.
* `qq export.sqlite` → safe backups. Recommend periodic `VACUUM` monthly.

---

## Security & Reliability

* DB path under `~/.qq/qq.db` with `0600` permissions.
* No network service exposed; all local process; least moving parts.
* WAL mode + periodic checkpoint to keep file size in check.

---

## Rollout Plan (Phased)

1. **Prototype (D1)**: Implement `db.init`, `ingest`, `query` in a branch; seed 10 docs; run eval.
2. **Migration (D2)**: `migrate.qdrant` + freeze Qdrant container; re‑run eval A/B.
3. **Flip (D3)**: Remove Qdrant codepath; ship v1.0 of sqlite backend.
4. **Polish (D4)**: Optional reranker; docs; CI.

---

## Agentic CLI Prompts (ready‑to‑run)

**Claude Code / Codex — Create store adapter**

```
You are coding inside the qq repo. Task: implement src/qq/store_sqlite.py providing SqliteVecStore with methods init/load_vec0/upsert/search/rebuild_fts/backup/doctor as per the plan below. Ensure SQLite tables (docs, docs_fts, vec_index) exist; detect embed dim at first init; enable WAL; write hybrid search (vector + BM25 + alpha fusion). Output: a working module and tests under tests/test_store_sqlite.py covering CRUD, hybrid fusion, and FTS rebuild.
```

**Claude Code — Wire CLI**

```
Add Typer commands: db.init, ingest, rebuild-fts, query, migrate.qdrant, doctor. JSONL output for query. Config at ~/.qq/config.yml. Use SentenceTransformers with normalize_embeddings=True.
```

**Codex — Qdrant migrator**

```
Implement qq/cli_migrate_qdrant.py: read points from Qdrant collection qq_data (batched), map payload {uri,title,meta,text} and vector to SQLite (docs, vec_index), then rebuild FTS. Idempotent: on conflict update docs and replace vec_index rowid.
```

---

## Acceptance Checklist

* [ ] `qq db.init` creates DB, sets correct `dim`.
* [ ] `qq ingest` handles txt/md/pdf, chunks, embeds, upserts.
* [ ] `qq query` returns JSONL + table; honors `--k`, `--alpha`.
* [ ] Hybrid only (no reranker) matches or beats Qdrant quality on gold set.
* [ ] Optional reranker improves MRR without >2× latency.
* [ ] `qq doctor` passes; `qq export.sqlite` produces a restorable file.
* [ ] Qdrant removed from repo; CI green.

---

## Risks & Mitigations

* **Embedding dim mismatch** → guard at `db.init`; store dim in `settings`; validate on ingest.
* **FTS external content drift** → `rebuild-fts` after bulk ops; unit tests.
* **Large PDFs creeping in** → chunk‑limit per ingest; warn when Nchunks > 200.
* **Reranker slowness** → keep disabled by default; only enable on demand.

---

## Next Steps (Actionable)

1. Implement `db.init`, `store_sqlite`, `search.hybrid` (T0–T3).
2. Wire Typer CLI; produce JSONL.
3. Port ingest to use new store; deprecate Qdrant code.
4. Build `migrate.qdrant`; run A/B eval; remove Docker.
5. Optional: add reranker flag + model.

> Ping me after T3; I’ll review the hybrid scoring and suggest tuning for your corpus.
