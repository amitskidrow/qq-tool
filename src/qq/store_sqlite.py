from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SearchResult:
    id: int
    score: float
    uri: str
    namespace: Optional[str]
    title: Optional[str]
    meta: Dict[str, Any]
    text: str


class SqliteVecStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    # --- Low-level helpers ---
    def load_vec0(self) -> None:
        try:
            import sqlite_vec  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sqlite-vec is required. Please install the 'sqlite-vec' package"
            ) from e
        # Enable extension loading if supported; required on some Python builds
        try:
            self.conn.enable_load_extension(True)  # type: ignore[attr-defined]
        except Exception:
            # If unavailable, sqlite_vec.load may still work if it uses connection hooks
            pass
        try:
            sqlite_vec.load(self.conn)
        except Exception as e:
            # Provide clearer guidance for the common 'not authorized' error when
            # extension loading is disabled at the connection level.
            msg = str(e)
            if "not authorized" in msg.lower():
                raise RuntimeError("sqlite-vec extension load failed: not authorized (enable extension loading)") from e
            raise

    def exec(self, sql: str, args: Tuple | Sequence = ()):
        return self.conn.execute(sql, args)

    def executemany(self, sql: str, seq: Iterable[Tuple | Sequence]):
        return self.conn.executemany(sql, seq)

    def commit(self) -> None:
        self.conn.commit()

    # --- Schema / lifecycle ---
    def ensure_schema(self, dim: int) -> None:
        self.load_vec0()
        # Create or migrate docs table to allow non-unique URIs (multiple chunks per source)
        # 1) Create table if missing (uri is NOT UNIQUE)
        self.exec(
            """
            CREATE TABLE IF NOT EXISTS docs (
              id INTEGER PRIMARY KEY,
              uri TEXT,
              namespace TEXT,
              title TEXT,
              meta TEXT,
              text TEXT,
              hash TEXT,
              simhash INTEGER,
              created_at TEXT,
              updated_at TEXT
            );
            """
        )
        # 2) If existing table had UNIQUE(uri), migrate to non-unique
        try:
            row = self.exec(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='docs'"
            ).fetchone()
            ddl = (row[0] or "") if row else ""
            if "uri TEXT UNIQUE" in ddl:
                # Perform table rebuild migration
                self.exec("BEGIN")
                self.exec("ALTER TABLE docs RENAME TO docs_old")
                self.exec(
                    """
                    CREATE TABLE docs (
                      id INTEGER PRIMARY KEY,
                      uri TEXT,
                      namespace TEXT,
                      title TEXT,
                      meta TEXT,
                      text TEXT,
                      hash TEXT,
                      simhash INTEGER,
                      created_at TEXT,
                      updated_at TEXT
                    );
                    """
                )
                # Recreate data preserving ids
                self.exec(
                    "INSERT INTO docs(id, uri, namespace, title, meta, text, hash, simhash, created_at, updated_at)\n                     SELECT id, uri, namespace, title, meta, text, hash, simhash, created_at, updated_at FROM docs_old"
                )
                # Drop old table
                self.exec("DROP TABLE docs_old")
                self.exec("COMMIT")
        except Exception:
            # If migration fails, try to ROLLBACK and continue; downstream ops may still work
            try:
                self.exec("ROLLBACK")
            except Exception:
                pass
        # FTS table with explicit rowid mapping; we will manage it manually on upserts/deletes
        self.exec(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(text, title, content='');
            """
        )
        # sqlite-vec index
        self.exec(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
            USING vec0(embedding float[{dim}]);
            """
        )
        # Ensure a non-unique index on uri exists (replace legacy unique index if needed)
        try:
            idx_list = self.exec("PRAGMA index_list('docs')").fetchall()
            needs_drop = any((r[1] == 'idx_docs_uri' and int(r[2]) == 1) for r in idx_list)  # unique==1
            if needs_drop:
                self.exec("DROP INDEX IF EXISTS idx_docs_uri")
        except Exception:
            pass
        self.exec("CREATE INDEX IF NOT EXISTS idx_docs_uri ON docs(uri)")
        self.exec(
            "CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT)"
        )
        self.exec(
            "INSERT OR REPLACE INTO settings(k,v) VALUES('dim', ?)", (str(dim),)
        )
        self.commit()

    def get_dim(self) -> Optional[int]:
        try:
            row = self.exec("SELECT v FROM settings WHERE k='dim'").fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    # --- CRUD ---
    def upsert_chunk(
        self,
        *,
        uri: str,
        namespace: str,
        text: str,
        vector: np.ndarray,
        title: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        hash_: Optional[str] = None,
        simhash: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> int:
        # Insert-or-replace docs row by unique (uri, text, namespace, hash?) semantics.
        # Our policy: URI can repeat across multiple chunks; we distinguish by row content.
        # Use a simple insert; duplicates will create multiple rows. Replace-mode handled by callers.
        cur = self.exec(
            """
            INSERT INTO docs(uri, namespace, title, meta, text, hash, simhash, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                uri,
                namespace,
                title,
                json.dumps(meta or {}, ensure_ascii=False),
                text,
                hash_,
                int(simhash) if simhash is not None else None,
                created_at,
                updated_at,
            ),
        )
        doc_id = int(cur.lastrowid)
        # FTS rowid mirrors docs.id
        self.exec(
            "INSERT INTO docs_fts(rowid, text, title) VALUES(?,?,?)",
            (doc_id, text, title or ""),
        )
        # Vector
        self.exec(
            "INSERT INTO vec_index(rowid, embedding) VALUES(?, ?)",
            (doc_id, json.dumps(vector.tolist())),
        )
        self.commit()
        return doc_id

    def delete_ids(self, ids: Sequence[int]) -> int:
        if not ids:
            return 0
        qmarks = ",".join(["?"] * len(ids))
        self.exec(f"DELETE FROM docs WHERE id IN ({qmarks})", tuple(ids))
        self.exec(f"DELETE FROM docs_fts WHERE rowid IN ({qmarks})", tuple(ids))
        self.exec(f"DELETE FROM vec_index WHERE rowid IN ({qmarks})", tuple(ids))
        self.commit()
        return len(ids)

    def list_by_uri(self, uri: str) -> List[sqlite3.Row]:
        return list(self.exec("SELECT * FROM docs WHERE uri=? LIMIT 100", (uri,)))

    def list_namespace(self, namespace: str, limit: int = 1000) -> List[sqlite3.Row]:
        return list(
            self.exec(
                "SELECT * FROM docs WHERE namespace=? LIMIT ?", (namespace, limit)
            )
        )

    # --- Search ---
    def search_dense(
        self, query_vec: np.ndarray, namespace: Optional[str] = None, topk: int = 8
    ) -> List[SearchResult]:
        payload: Tuple[Any, ...]
        if namespace:
            rows = self.exec(
                """
                SELECT d.id, d.uri, d.namespace, d.title, d.meta, d.text, 1.0 - v.distance AS sim
                FROM vec_index v JOIN docs d ON d.id = v.rowid
                WHERE d.namespace = ? AND v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (
                    namespace,
                    json.dumps(query_vec.tolist()),
                    int(max(1, topk)),
                ),
            ).fetchall()
        else:
            rows = self.exec(
                """
                SELECT d.id, d.uri, d.namespace, d.title, d.meta, d.text, 1.0 - v.distance AS sim
                FROM vec_index v JOIN docs d ON d.id = v.rowid
                WHERE v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (
                    json.dumps(query_vec.tolist()),
                    int(max(1, topk)),
                ),
            ).fetchall()
        out: List[SearchResult] = []
        for r in rows:
            out.append(
                SearchResult(
                    id=int(r["id"]),
                    score=float(r["sim"]),
                    uri=r["uri"],
                    namespace=r["namespace"],
                    title=r["title"],
                    meta=json.loads(r["meta"] or "{}"),
                    text=r["text"] or "",
                )
            )
        return out

    def search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text: str,
        namespace: Optional[str] = None,
        topk: int = 8,
        alpha: float = 0.55,
        pool: int = 200,
    ) -> List[SearchResult]:
        # Dense candidates
        if namespace:
            vec_rows = self.exec(
                """
                SELECT d.id, 1.0 - v.distance AS sim
                FROM vec_index v JOIN docs d ON d.id=v.rowid
                WHERE d.namespace=? AND v.embedding MATCH ?
                ORDER BY v.distance LIMIT ?
                """,
                (
                    namespace,
                    json.dumps(query_vec.tolist()),
                    int(max(topk * 2, 20)),
                ),
            ).fetchall()
        else:
            vec_rows = self.exec(
                """
                SELECT d.id, 1.0 - v.distance AS sim
                FROM vec_index v JOIN docs d ON d.id=v.rowid
                WHERE v.embedding MATCH ?
                ORDER BY v.distance LIMIT ?
                """,
                (
                    json.dumps(query_vec.tolist()),
                    int(max(topk * 2, 20)),
                ),
            ).fetchall()

        # Lexical via FTS5
        if namespace:
            bm_rows = self.exec(
                """
                SELECT d.id, bm25(docs_fts) AS s
                FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
                WHERE d.namespace=? AND docs_fts MATCH ?
                ORDER BY s LIMIT ?
                """,
                (namespace, query_text, int(max(topk * 2, min(20, pool)))),
            ).fetchall()
        else:
            bm_rows = self.exec(
                """
                SELECT d.id, bm25(docs_fts) AS s
                FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
                WHERE docs_fts MATCH ?
                ORDER BY s LIMIT ?
                """,
                (query_text, int(max(topk * 2, min(20, pool)))),
            ).fetchall()

        # Normalize and fuse
        bm = {int(i): float(s) for i, s in bm_rows}
        if bm:
            scores = np.array(list(bm.values()), dtype=np.float32)
            lo, hi = float(scores.min()), float(scores.max())
            rng = (hi - lo) or 1.0
            for i in bm:
                bm[i] = 1.0 - ((bm[i] - lo) / rng)

        sim = {int(i): float(s) for i, s in vec_rows}
        cand = set(sim) | set(bm)
        fused = [
            (i, (alpha * sim.get(i, 0.0)) + ((1.0 - alpha) * bm.get(i, 0.0)))
            for i in cand
        ]
        fused.sort(key=lambda t: t[1], reverse=True)
        top_ids = [i for i, _ in fused[:topk]]
        if not top_ids:
            return []

        qmarks = ",".join(["?"] * len(top_ids))
        rows = self.exec(
            f"SELECT id, uri, namespace, title, meta, text FROM docs WHERE id IN ({qmarks})",
            tuple(top_ids),
        ).fetchall()
        out: List[SearchResult] = []
        by_id = {int(r["id"]): r for r in rows}
        for i in top_ids:
            r = by_id.get(i)
            if not r:
                continue
            out.append(
                SearchResult(
                    id=int(r["id"]),
                    score=float(next((s for j, s in fused if j == i), 0.0)),
                    uri=r["uri"],
                    namespace=r["namespace"],
                    title=r["title"],
                    meta=json.loads(r["meta"] or "{}"),
                    text=r["text"] or "",
                )
            )
        return out

    # --- Maintenance ---
    def rebuild_fts(self) -> int:
        self.exec("DELETE FROM docs_fts")
        rows = self.exec("SELECT id, text, title FROM docs").fetchall()
        self.executemany(
            "INSERT INTO docs_fts(rowid, text, title) VALUES(?,?,?)",
            [(int(r["id"]), r["text"] or "", r["title"] or "") for r in rows],
        )
        self.commit()
        return len(rows)

    def backup(self, dest_path: Path | str) -> Path:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Ensure pending writes are flushed
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.commit()
        shutil.copy2(self.db_path, dest)
        # Copy -wal if exists
        wal = self.db_path.with_suffix(self.db_path.suffix + "-wal")
        if wal.exists():
            try:
                shutil.copy2(wal, dest.with_suffix(dest.suffix + "-wal"))
            except Exception:
                pass
        return dest

    def doctor(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": True, "checks": []}
        def check(name: str, ok: bool, detail: Optional[str] = None):
            out["checks"].append({"name": name, "ok": ok, "detail": detail})
            if not ok:
                out["ok"] = False

        # Tables
        try:
            t = set(
                r[0]
                for r in self.exec(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view','virtual table')"
                ).fetchall()
            )
            need = {"docs", "docs_fts", "vec_index", "settings"}
            check("tables_present", need.issubset(t), f"have={sorted(t)}")
        except Exception as e:
            check("tables_present", False, str(e))

        # Dim
        try:
            dim_row = self.exec("SELECT v FROM settings WHERE k='dim'").fetchone()
            check("dimension_set", bool(dim_row and dim_row[0]), f"dim={dim_row[0] if dim_row else None}")
        except Exception as e:
            check("dimension_set", False, str(e))

        # Orphans
        try:
            c_docs = self.exec("SELECT COUNT(*) FROM docs").fetchone()[0]
            c_vec = self.exec("SELECT COUNT(*) FROM vec_index").fetchone()[0]
            c_fts = self.exec("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
            ok = (c_docs == c_vec == c_fts)
            check("orphan_consistency", ok, f"docs={c_docs}, vec={c_vec}, fts={c_fts}")
        except Exception as e:
            check("orphan_consistency", False, str(e))

        return out
