from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable, List

import numpy as np

from .hashing import content_hash
from .tokens import rough_token_count
from .qq_embeddings import get_embedder


DEFAULT_INDEX_DB = os.path.expanduser("~/.qq/index.sqlite")


def _utc_now_iso() -> str:
    # Use Zulu ISO-8601 timestamps for portability
    import datetime as _dt

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class IngestCounts:
    docs: int = 0
    chunks: int = 0
    duplicates: int = 0
    embedded: int = 0
    nodes: int = 0
    edges: int = 0


class Store:
    """
    SQLite store for new-arch pipeline.

    Schema (subset of PRD v1):
      - docs(doc_id, title, type, source, created_at, updated_at, tags)
      - chunks(chunk_id, doc_id, section_path, kind, content, token_count, created_at, updated_at, content_hash)
      - fts(content, section_path, doc_id UNINDEXED, chunk_id UNINDEXED)
      - vec (sqlite-vec), with auxiliary mapping table vec_map(chunk_id -> rowid)
      - nodes/edges (created but unused in M1)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.getenv("QQ_INDEX_DB", DEFAULT_INDEX_DB)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        self._vec_enabled = self._try_load_vec()
        self._init_schema()
        # Lazy embedder init on demand
        self._embedder = None

    # ---------- setup ----------
    def _try_load_vec(self) -> bool:
        try:
            self.conn.enable_load_extension(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            import sqlite_vec  # type: ignore

            sqlite_vec.load(self.conn)
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        with self.conn:
            # docs table (non-conflicting name with legacy engine by using different DB file)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs(
                  doc_id TEXT PRIMARY KEY,
                  title TEXT,
                  type TEXT,
                  source TEXT,
                  created_at TEXT,
                  updated_at TEXT,
                  tags TEXT
                )
                """
            )
            # chunks table with content_hash for dedup
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks(
                  chunk_id INTEGER PRIMARY KEY,
                  doc_id TEXT NOT NULL,
                  section_path TEXT,
                  kind TEXT CHECK(kind IN ('prose','code','workflow','feedback')),
                  content TEXT NOT NULL,
                  token_count INT,
                  content_hash TEXT,
                  created_at TEXT,
                  updated_at TEXT,
                  FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
                )
                """
            )
            # Uniqueness guard for incremental ingest
            self.conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uniq_chunk ON chunks(doc_id, section_path, content_hash)
                """
            )
            # FTS5 virtual table
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
                  content, section_path, doc_id UNINDEXED, chunk_id UNINDEXED,
                  tokenize='unicode61'
                )
                """
            )
            # sqlite-vec table (dimension decided at first embed)
            if self._vec_enabled:
                # Create a generic vec table; dimension enforced by inserted vectors
                try:
                    self.conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(embedding float[384])"
                    )
                except Exception:
                    # Some builds require dynamic dims; skip creation until first insert
                    pass
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vec_map(
                      chunk_id INTEGER PRIMARY KEY,
                      rid INTEGER UNIQUE
                    )
                    """
                )
            # Graph-lite tables (not populated in M1)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes(
                  node_id INTEGER PRIMARY KEY,
                  kind TEXT,
                  name TEXT,
                  ref_chunk_id INT,
                  extra TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges(
                  src INT, dst INT, rel TEXT, weight REAL DEFAULT 1.0,
                  PRIMARY KEY (src, dst, rel)
                )
                """
            )
            # FTS for nodes name lookups
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                  name, node_id UNINDEXED
                )
                """
            )


    # ---------- public API ----------
    def upsert_doc(self, *, doc_id: str, title: Optional[str], type_: str, source: str, tags: Optional[Dict[str, Any]] = None) -> None:
        now = _utc_now_iso()
        tags_json = json.dumps(tags) if tags is not None else None
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO docs(doc_id, title, type, source, created_at, updated_at, tags)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  title=excluded.title,
                  type=excluded.type,
                  source=excluded.source,
                  updated_at=excluded.updated_at,
                  tags=excluded.tags
                """,
                (doc_id, title, type_, source, now, now, tags_json),
            )

    def insert_chunk(self, *, doc_id: str, section_path: Optional[str], kind: str, content: str) -> Tuple[Optional[int], bool]:
        """
        Insert chunk and mirror into FTS. Returns (chunk_id, is_duplicate).
        """
        ch = content_hash(content)
        now = _utc_now_iso()
        tok = rough_token_count(content)
        # Check duplicate against (doc_id, section_path, hash)
        cur = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=? AND (section_path IS ? OR section_path=?) AND content_hash=?",
            (doc_id, section_path, section_path, ch),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0]), True
        with self.conn:
            cur2 = self.conn.execute(
                """
                INSERT INTO chunks(doc_id, section_path, kind, content, token_count, content_hash, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (doc_id, section_path, kind, content, tok, ch, now, now),
            )
            cid = int(cur2.lastrowid)
            # Mirror into FTS (idempotent ensured by delete+insert pattern if needed)
            self.conn.execute("INSERT INTO fts(content, section_path, doc_id, chunk_id) VALUES(?,?,?,?)", (content, section_path, doc_id, cid))
        return cid, False

    def insert_vec(self, chunk_id: int, embedding: np.ndarray) -> None:
        if not self._vec_enabled:
            return
        # Ensure 1-D float32
        vec = embedding.astype(np.float32).reshape(-1)
        # Try to ensure vec table exists with the correct dim if not already created
        dim = int(vec.shape[0])
        try:
            self.conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(embedding float[{dim}])")
        except Exception:
            pass
        # Serialize using sqlite-vec helpers when available
        try:
            import sqlite_vec  # type: ignore

            if hasattr(sqlite_vec, "serialize"):
                blob = sqlite_vec.serialize(vec.tolist())  # type: ignore
            elif hasattr(sqlite_vec, "serialize_float32"):
                blob = sqlite_vec.serialize_float32(vec)  # type: ignore
            elif hasattr(sqlite_vec, "pack"):
                blob = sqlite_vec.pack(vec.tolist())  # type: ignore
            else:
                blob = sqlite3.Binary(vec.tobytes())
        except Exception:
            blob = sqlite3.Binary(vec.tobytes())

        with self.conn:
            cur = self.conn.execute("INSERT INTO vec(embedding) VALUES(?)", (blob,))
            rid = int(cur.lastrowid)
            self.conn.execute(
                "INSERT OR REPLACE INTO vec_map(chunk_id, rid) VALUES(?,?)",
                (chunk_id, rid),
            )

    # ---------- helpers ----------
    def stats(self) -> Dict[str, int]:
        q = {
            "docs": "SELECT COUNT(1) FROM docs",
            "chunks": "SELECT COUNT(1) FROM chunks",
            "fts": "SELECT COUNT(1) FROM fts",
            "vec": "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='vec'",
            "nodes": "SELECT COUNT(1) FROM nodes",
            "edges": "SELECT COUNT(1) FROM edges",
        }
        out: Dict[str, int] = {}
        for k, sql in q.items():
            try:
                row = self.conn.execute(sql).fetchone()
                out[k] = int(row[0]) if row is not None else 0
            except Exception:
                out[k] = 0
        return out

    def export_to(self, dest_path: str) -> str:
        if not dest_path:
            raise ValueError("destination path required")
        dest = sqlite3.connect(dest_path)
        try:
            self.conn.backup(dest)
        finally:
            dest.close()
        return dest_path

    def import_from(self, src_path: str) -> None:
        if not src_path or not Path(src_path).exists():
            raise ValueError(f"source not found: {src_path}")
        # Replace current DB by copying over
        src = sqlite3.connect(src_path)
        try:
            with self.conn:
                # Clear current DB by recreating schema after wipe
                self.conn.execute("PRAGMA writable_schema = 1;")
                self.conn.execute("DELETE FROM sqlite_master WHERE type IN ('table','index','trigger');")
                self.conn.execute("PRAGMA writable_schema = 0;")
                self.conn.execute("VACUUM;")
            # Restore from src
            self.conn.close()
            Path(self.db_path).unlink(missing_ok=True)
            # Simpler: copy file bytes
            import shutil as _sh

            _sh.copy2(src_path, self.db_path)
            # Reconnect
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        finally:
            src.close()


    # ----- graph-lite -----
    def upsert_node(self, *, kind: str, name: str, ref_chunk_id: int | None = None, extra: dict | None = None) -> int:
        extra_json = json.dumps(extra) if extra is not None else None
        row = self.conn.execute(
            "SELECT node_id FROM nodes WHERE kind=? AND name=? AND (ref_chunk_id IS ? OR ref_chunk_id=?)",
            (kind, name, ref_chunk_id, ref_chunk_id),
        ).fetchone()
        if row is not None:
            node_id = int(row[0])
        else:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO nodes(kind, name, ref_chunk_id, extra) VALUES(?,?,?,?)",
                    (kind, name, ref_chunk_id, extra_json),
                )
                node_id = int(cur.lastrowid)
        try:
            with self.conn:
                self.conn.execute("INSERT INTO nodes_fts(name, node_id) VALUES(?, ?)", (name, node_id))
        except Exception:
            pass
        return node_id

    def build_graph_for_doc(self, doc_id: str) -> dict:
        rows = self.conn.execute(
            "SELECT chunk_id, section_path FROM chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()
        created = 0
        with self.conn:
            for r in rows:
                cid = int(r[0])
                sp = r[1] if r[1] is not None else None
                if not sp:
                    continue
                parts = [seg.strip() for seg in str(sp).split('>') if seg.strip()]
                prev_node = None
                for seg in parts:
                    nid = self.upsert_node(kind='concept', name=seg, ref_chunk_id=cid, extra=None)
                    created += 1
                    if prev_node is not None:
                        try:
                            self.conn.execute(
                                "INSERT OR IGNORE INTO edges(src, dst, rel, weight) VALUES(?,?,?,?)",
                                (prev_node, nid, 'contains', 1.0),
                            )
                        except Exception:
                            pass
                    prev_node = nid
        return {"ok": True, "created": created}

    def _graph_match_chunk_ids(self, q: str) -> set[int]:
        import re as _re
        toks = _re.findall(r"\w+", q, flags=_re.UNICODE)[:16]
        if not toks:
            return set()
        expr = " OR ".join(f'"{t}"' for t in toks)
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT n.ref_chunk_id FROM nodes n JOIN nodes_fts f ON f.node_id = n.node_id WHERE nodes_fts MATCH ?",
                (expr,),
            ).fetchall()
            return {int(r[0]) for r in rows if r[0] is not None}
        except Exception:
            return set()

    def _should_graph_boost(self, q: str, force: bool) -> bool:
        if force:
            return True
        import re as _re
        return bool(_re.search(r"\b(depends on|calls|before|after|owner|implements)\b", q, flags=_re.I))

    def _apply_graph_boost(self, scores: dict[int, float], G: set[int], boost: float = 0.05, cap: float = 0.10) -> None:
        for cid in list(scores.keys()):
            if cid in G:
                scores[cid] = scores[cid] + min(boost, cap)

    # ---------- retrieval ----------
    def _pack_vec(self, v: np.ndarray):
        try:
            import sqlite_vec  # type: ignore

            vec = v.astype(np.float32).reshape(-1)
            if hasattr(sqlite_vec, "serialize"):
                return sqlite_vec.serialize(vec.tolist())  # type: ignore
            if hasattr(sqlite_vec, "serialize_float32"):
                return sqlite_vec.serialize_float32(vec)  # type: ignore
            if hasattr(sqlite_vec, "pack"):
                return sqlite_vec.pack(vec.tolist())  # type: ignore
            return sqlite3.Binary(vec.tobytes())
        except Exception:
            return sqlite3.Binary(v.astype(np.float32).reshape(-1).tobytes())

    @property
    def vec_enabled(self) -> bool:
        return bool(self._vec_enabled)

    def _safe_fts_query(self, q: str) -> Optional[str]:
        import re as _re

        toks = _re.findall(r"\w+", q, flags=_re.UNICODE)
        toks = [t for t in toks if t]
        if not toks:
            return None
        toks = toks[:32]
        return " OR ".join(f'"{t}"' for t in toks)

    def search_fts(self, q: str, k: int) -> List[Tuple[int, float]]:
        expr = self._safe_fts_query(q)
        if not expr:
            return []
        cur = self.conn.execute(
            "SELECT chunk_id, bm25(fts) AS rank FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
            (expr, k),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        ranks = [float(r["rank"]) if "rank" in r.keys() else float(r[1]) for r in rows]
        max_r = max(ranks) if ranks else 1.0
        min_r = min(ranks) if ranks else 0.0
        denom = max(max_r - min_r, 1e-9)
        out: List[Tuple[int, float]] = []
        for r in rows:
            cid = int(r["chunk_id"]) if "chunk_id" in r.keys() else int(r[0])
            val = float(r["rank"]) if "rank" in r.keys() else float(r[1])
            sim = 1.0 - ((val - min_r) / denom)
            out.append((cid, sim))
        return out

    def search_vec(self, q: str, k: int) -> List[Tuple[int, float]]:
        if not self._vec_enabled:
            return []
        # Init embedder lazily
        if self._embedder is None:
            self._embedder = get_embedder()
        q_vec = self._embedder.encode([q])
        blob = self._pack_vec(q_vec)
        cur = self.conn.execute(
            "SELECT rowid, distance FROM vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, k),
        )
        out: List[Tuple[int, float]] = []
        for r in cur.fetchall():
            rid = int(r["rowid"]) if "rowid" in r.keys() else int(r[0])
            dist = float(r["distance"]) if "distance" in r.keys() else float(r[1])
            sim = 1.0 - dist
            id_row = self.conn.execute("SELECT chunk_id FROM vec_map WHERE rid=?", (rid,)).fetchone()
            if id_row:
                out.append((int(id_row[0]), sim))
        return out

    def search_hybrid(
        self,
        q: str,
        *,
        K_lex: int = 30,
        K_sem: int = 30,
        K_merge: int = 50,
        alpha: float = 0.55,
        graph_boost: bool = False,
    ) -> List[Tuple[int, float]]:
        dense: List[Tuple[int, float]] = []
        if self._vec_enabled:
            try:
                dense = self.search_vec(q, K_sem)
            except Exception:
                dense = []
        sparse = self.search_fts(q, K_lex)
        scores: Dict[int, float] = {}
        if dense:
            for cid, s in dense:
                scores[cid] = scores.get(cid, 0.0) + alpha * s
        if sparse:
            w = (1.0 - alpha) if self._vec_enabled else 1.0
            for cid, s in sparse:
                scores[cid] = scores.get(cid, 0.0) + w * s
        # Graph-lite boost
        try:
            if self._should_graph_boost(q, graph_boost):
                G = self._graph_match_chunk_ids(q)
                if G:
                    self._apply_graph_boost(scores, G, 0.05, 0.10)
        except Exception:
            pass
        items = list(scores.items())
        items.sort(key=lambda t: t[1], reverse=True)
        return items[:K_merge]

    def get_chunks(self, ids: Iterable[int]) -> Dict[str, Dict[str, Any]]:
        ids = list({int(x) for x in ids})
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        sql = f"""
        SELECT c.chunk_id, c.content, c.section_path, d.doc_id, d.source
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.chunk_id IN ({qmarks})
        """
        rows = self.conn.execute(sql, ids).fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            cid = int(r["chunk_id"]) if "chunk_id" in r.keys() else int(r[0])
            content = r["content"] if "content" in r.keys() else r[1]
            section_path = r["section_path"] if "section_path" in r.keys() else r[2]
            doc_id = r["doc_id"] if "doc_id" in r.keys() else r[3]
            source = r["source"] if "source" in r.keys() else r[4]
            out[str(cid)] = {"text": content, "path": source, "section_path": section_path, "doc_id": doc_id}
        return out


def ingest_text(
    store: Store,
    *,
    doc_id: str,
    text: str,
    title: Optional[str] = None,
    kind: str = "prose",
    section_path: Optional[str] = None,
    source: str = "local",
    tags: Optional[Dict[str, Any]] = None,
    embed: bool = True,
) -> IngestCounts:
    """
    Minimal ingest: upsert doc, chunk text, add to FTS, embed chunks.
    """
    counts = IngestCounts()
    store.upsert_doc(doc_id=doc_id, title=title, type_=kind, source=source, tags=tags)
    counts.docs += 1

    # naive chunking fallback; callers can pre-chunk if they wish
    from .chunking import simple_chunks

    chunks = simple_chunks(text, max_chars=1200, overlap=120)
    emb = get_embedder() if embed else None

    for ch_text in chunks:
        cid, dup = store.insert_chunk(
            doc_id=doc_id,
            section_path=section_path,
            kind=kind,
            content=ch_text,
        )
        if dup:
            counts.duplicates += 1
            continue
        counts.chunks += 1
        if embed and emb is not None:
            vec = emb.encode([ch_text])[0]
            store.insert_vec(cid, vec)  # type: ignore[arg-type]
            counts.embedded += 1
    return counts
