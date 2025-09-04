from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import sqlite_vec  # type: ignore
except Exception as e:  # pragma: no cover
    sqlite_vec = None  # type: ignore

from .qq_embeddings import OnnxEmbedder, get_embedder


DEFAULT_DB_URI = os.path.expanduser("~/.qq/global.db")


@dataclass
class SearchHit:
    id: str
    score: float


@dataclass
class Timings:
    embed_ms: float = 0.0
    vec_ms: float = 0.0
    fts_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


class Engine:
    """
    In-process retrieval engine: in-memory SQLite + sqlite-vec + FTS5.

    - All runtime data lives only in RAM using URI: file:qqmem?mode=memory&cache=shared
    - Vector similarity via sqlite-vec (exact KNN on normalized embeddings)
    - Keyword search via FTS5 (BM25)
    - Fusion weighted by alpha
    """

    def __init__(
        self,
        *,
        db_uri: str = DEFAULT_DB_URI,
        embedder: Optional[OnnxEmbedder] = None,
        dim: Optional[int] = None,
    ) -> None:
        # Resolve DB location (global persistent by default)
        self.db_uri = db_uri or DEFAULT_DB_URI
        if not self.db_uri:
            self.db_uri = DEFAULT_DB_URI
        # Ensure parent directory exists for file-based DBs
        try:
            if not self.db_uri.startswith("file:"):
                Path(self.db_uri).parent.mkdir(parents=True, exist_ok=True)
            else:
                # file: URI may contain absolute path after the scheme
                p = self.db_uri.removeprefix("file:")
                # Strip URI query string if present
                p = p.split("?", 1)[0]
                if p:
                    Path(p).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort, continue if path cannot be created
            pass
        # Defer embedder creation until after connection (so we know if vec is enabled)
        self._embedder = None
        self.dim = dim or 384  # default; will adjust if embedder is created
        self._lock = threading.RLock()
        self._conn = self._connect()
        self.vec_enabled = False
        self._init_db()
        # Initialize embedder only if vectors are enabled and not supplied
        if self.vec_enabled:
            self._embedder = embedder or get_embedder()
            # Update dim if not provided
            if dim is None and hasattr(self._embedder, "dim"):
                self.dim = int(self._embedder.dim)

    def _connect(self) -> sqlite3.Connection:
        is_uri = isinstance(self.db_uri, str) and self.db_uri.startswith("file:")
        conn = sqlite3.connect(self.db_uri, uri=is_uri, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable extension loading and FTS5
        try:
            # Required to allow loading loadable extensions like sqlite-vec
            try:
                conn.enable_load_extension(True)
            except Exception:
                pass
            if sqlite_vec is not None:
                try:
                    sqlite_vec.load(conn)
                    self.vec_enabled = True
                except Exception:
                    # Could not load vec extension; fall back to FTS-only
                    self.vec_enabled = False
        except Exception:
            # Extension loading failed entirely; continue with FTS-only
            self.vec_enabled = False

        # Prefer WAL for durability/perf on file-backed DBs
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._conn:  # implicit transaction
            # Main document store
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                  id TEXT PRIMARY KEY,
                  text TEXT NOT NULL,
                  meta TEXT
                )
                """
            )
            # FTS5 table for BM25; 'id' is stored but not indexed
            # Use default tokenizer for maximum compatibility across builds
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                  id UNINDEXED,
                  text
                )
                """
            )
            # Vector index using sqlite-vec (optional)
            if self.vec_enabled:
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(embedding float[{self.dim}])"
                )

    # ---------- helpers ----------
    def _pack(self, v: np.ndarray) -> memoryview:
        # Serialize a 1-D float32 vector for sqlite-vec
        if sqlite_vec is None:  # pragma: no cover
            raise RuntimeError("sqlite-vec Python package not available")
        vec = v.astype(np.float32)
        # Support multiple sqlite-vec APIs across versions
        if hasattr(sqlite_vec, "serialize"):
            return sqlite_vec.serialize(vec.tolist())  # type: ignore
        if hasattr(sqlite_vec, "serialize_float32"):
            return sqlite_vec.serialize_float32(vec)  # type: ignore
        if hasattr(sqlite_vec, "pack"):
            # 'pack' may expect a Python list of floats
            return sqlite_vec.pack(vec.tolist())  # type: ignore
        # Fallback: raw bytes wrapped as memoryview
        return memoryview(vec.tobytes())

    # ---------- public API ----------
    def upsert(self, id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Embed (only if vectors enabled)
        vec = None
        e0 = e1 = time.perf_counter()
        if self.vec_enabled and self._embedder is not None:
            e0 = time.perf_counter()
            vec = self._embedder.encode([text])[0]
            e1 = time.perf_counter()

        with self._lock, self._conn:  # single transaction
            # docs
            self._conn.execute(
                "INSERT INTO docs(id, text, meta) VALUES(?, ?, ?)\n"
                "ON CONFLICT(id) DO UPDATE SET text=excluded.text, meta=excluded.meta",
                (id, text, json.dumps(meta) if meta is not None else None),
            )
            # FTS (replace via delete+insert — FTS5 doesn't support ON CONFLICT)
            self._conn.execute("DELETE FROM docs_fts WHERE id = ?", (id,))
            self._conn.execute(
                "INSERT INTO docs_fts(id, text) VALUES(?, ?)",
                (id, text),
            )
            # Vectors (optional): maintain mapping only if vec is enabled
            if self.vec_enabled and vec is not None:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vec_map (
                      id TEXT PRIMARY KEY,
                      rid INTEGER UNIQUE
                    )
                    """
                )
                cur = self._conn.execute("SELECT rid FROM vec_map WHERE id = ?", (id,))
                row = cur.fetchone()
                if row is None:
                    # Insert new vector row
                    cur2 = self._conn.execute(
                        "INSERT INTO vec(embedding) VALUES(?)",
                        (self._pack(vec),),
                    )
                    rid = cur2.lastrowid
                    self._conn.execute(
                        "INSERT INTO vec_map(id, rid) VALUES(?, ?)", (id, rid)
                    )
                else:
                    rid = int(row["rid"])
                    self._conn.execute(
                        "UPDATE vec SET embedding=? WHERE rowid=?",
                        (self._pack(vec), rid),
                    )

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": True,
            "timings": {
                "embed_ms": max((e1 - e0) * 1000.0, 0.0),
                "total_ms": total_ms,
            },
        }

    def _search_vec(self, q_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        # Returns list of (id, sim) where sim in [0,1], higher is better
        if not self.vec_enabled:
            return []
        if q_vec.ndim == 2:
            q_vec = q_vec[0]
        cur = self._conn.execute(
            "SELECT rowid, distance FROM vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (self._pack(q_vec), k),
        )
        out: List[Tuple[str, float]] = []
        for r in cur.fetchall():
            rid = int(r["rowid"]) if "rowid" in r.keys() else int(r[0])
            dist = float(r["distance"]) if "distance" in r.keys() else float(r[1])
            # Convert cosine distance -> similarity
            sim = 1.0 - dist
            id_row = self._conn.execute(
                "SELECT id FROM vec_map WHERE rid = ?", (rid,)
            ).fetchone()
            if id_row:
                out.append((id_row[0], sim))
        return out

    def _search_fts(self, q: str, k: int) -> List[Tuple[str, float]]:
        # FTS5 BM25: smaller is better -> convert to similarity
        cur = self._conn.execute(
            "SELECT id, bm25(docs_fts) AS rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
            (q, k),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        ranks = [float(r["rank"]) if "rank" in r.keys() else float(r[1]) for r in rows]
        # Invert and normalize into [0,1]
        max_r = max(ranks) if ranks else 1.0
        min_r = min(ranks) if ranks else 0.0
        denom = max(max_r - min_r, 1e-9)
        out: List[Tuple[str, float]] = []
        for r in rows:
            rid = r["id"] if "id" in r.keys() else r[0]
            val = float(r["rank"]) if "rank" in r.keys() else float(r[1])
            sim = 1.0 - ((val - min_r) / denom)
            out.append((rid, sim))
        return out

    def query(
        self,
        q: str,
        *,
        k: int = 6,
        alpha: float = 0.55,
        rerank: bool = False,
    ) -> Tuple[List[SearchHit], Timings]:
        t0 = time.perf_counter()
        # Embed only if vectors are enabled
        e0 = e1 = time.perf_counter()
        dense: List[Tuple[str, float]] = []
        if self.vec_enabled and self._embedder is not None:
            e0 = time.perf_counter()
            q_vec = self._embedder.encode([q])
            e1 = time.perf_counter()
            # Vector search
            v0 = time.perf_counter()
            dense = self._search_vec(q_vec, k)
            v1 = time.perf_counter()
        else:
            v0 = v1 = time.perf_counter()

        # FTS search (always available)
        f0 = time.perf_counter()
        sparse = self._search_fts(q, k)
        f1 = time.perf_counter()

        # Fuse
        fu0 = time.perf_counter()
        scores: Dict[str, float] = {}
        if self.vec_enabled and dense:
            for id_, s in dense:
                scores[id_] = scores.get(id_, 0.0) + alpha * s
        # If no vectors, rely solely on FTS scores
        if sparse:
            weight = (1.0 - alpha) if self.vec_enabled else 1.0
            for id_, s in sparse:
                scores[id_] = scores.get(id_, 0.0) + weight * s
        hits = [SearchHit(id=kid, score=sv) for kid, sv in scores.items()]
        hits.sort(key=lambda x: x.score, reverse=True)
        hits = hits[:k]
        fu1 = time.perf_counter()

        rr_ms = 0.0
        # Optional rerank placeholder (off by default to maintain latency target)
        if rerank and hits:
            # No-op rerank to keep plumbing in place
            pass

        t1 = time.perf_counter()
        timings = Timings(
            embed_ms=max((e1 - e0) * 1000.0, 0.0),
            vec_ms=max((v1 - v0) * 1000.0, 0.0),
            fts_ms=(f1 - f0) * 1000.0,
            fuse_ms=(fu1 - fu0) * 1000.0,
            rerank_ms=rr_ms,
            total_ms=(t1 - t0) * 1000.0,
        )
        return hits, timings

    def snapshot(self, path: str) -> Dict[str, Any]:
        # Write a consistent snapshot to a file (absolute path recommended)
        if not path:
            raise ValueError("snapshot path required")
        with self._lock:
            dest = sqlite3.connect(path)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()
        return {"ok": True, "path": path}
