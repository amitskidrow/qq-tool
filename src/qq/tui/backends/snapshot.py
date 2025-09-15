from __future__ import annotations

import json
import sqlite3
from typing import Iterable, List, Optional

from ..models import DocDetail, DocRow, Page
from .base import Backend


class SnapshotBackend(Backend):
    def __init__(self, db_path: str) -> None:
        uri = f"file:{db_path}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def _safe_fts_query(self, q: str) -> Optional[str]:
        import re as _re

        toks = _re.findall(r"\w+", q, flags=_re.UNICODE)
        toks = [t for t in toks if t]
        if not toks:
            return None
        toks = toks[:32]
        return " OR ".join(f'"{t}"' for t in toks)

    async def list_docs(self, q: Optional[str], like: Optional[str], limit: int, offset: int) -> Page:
        rows: List[DocRow] = []
        if q:
            expr = self._safe_fts_query(q)
            if not expr:
                return {"rows": []}
            sql = (
                """
                SELECT d.doc_id AS id,
                       d.source AS uri,
                       d.title AS title,
                       d.updated_at AS updated,
                       (SELECT SUM(length(c2.content)) FROM chunks c2 WHERE c2.doc_id=d.doc_id) AS size,
                       MIN(bm25(fts)) AS bm25,
                       (SELECT COUNT(1) FROM chunks c3 WHERE c3.doc_id=d.doc_id) AS chunk_count,
                       (SELECT COALESCE(SUM(c4.token_count),0) FROM chunks c4 WHERE c4.doc_id=d.doc_id) AS token_count,
                       substr(d.tags, 1, 120) AS meta_summary
                FROM fts
                JOIN chunks c ON c.chunk_id = fts.chunk_id
                JOIN docs d ON d.doc_id = c.doc_id
                WHERE fts MATCH ?
                GROUP BY d.doc_id
                ORDER BY bm25 ASC
                LIMIT ? OFFSET ?
                """
            )
            cur = self.conn.execute(sql, (expr, int(limit), int(offset)))
            tmp = cur.fetchall()
            if not tmp:
                return {"rows": []}
            vals = [float(r["bm25"]) if "bm25" in r.keys() else float(r[5]) for r in tmp]
            max_r, min_r = max(vals), min(vals)
            denom = max(max_r - min_r, 1e-9)
            for r in tmp:
                bm25_val = float(r["bm25"]) if "bm25" in r.keys() else float(r[5])
                score = 1.0 - ((bm25_val - min_r) / denom)
                rows.append(
                    {
                        "id": r["id"],
                        "uri": r["uri"],
                        "title": r["title"],
                        "size": int(r["size"]) if r["size"] is not None else 0,
                        "updated": r["updated"],
                        "score": float(score),
                        "chunk_count": int(r["chunk_count"]) if r["chunk_count"] is not None else 0,
                        "token_count": int(r["token_count"]) if r["token_count"] is not None else 0,
                        "meta_summary": r["meta_summary"],
                    }
                )
            return {"rows": rows}
        # Recency or LIKE filter
        where = []
        params: List[object] = []
        if like:
            where.append("(d.source LIKE ? OR d.title LIKE ?)")
            params.extend([like, like])
        sql = (
            """
            SELECT d.doc_id AS id,
                   d.source AS uri,
                   d.title AS title,
                   d.updated_at AS updated,
                   (SELECT SUM(length(c2.content)) FROM chunks c2 WHERE c2.doc_id=d.doc_id) AS size,
                   (SELECT COUNT(1) FROM chunks c3 WHERE c3.doc_id=d.doc_id) AS chunk_count,
                   (SELECT COALESCE(SUM(c4.token_count),0) FROM chunks c4 WHERE c4.doc_id=d.doc_id) AS token_count,
                   substr(d.tags, 1, 120) AS meta_summary
            FROM docs d
            {where}
            ORDER BY d.updated_at DESC
            LIMIT ? OFFSET ?
            """
        ).format(where=("WHERE " + " AND ".join(where)) if where else "")
        params.extend([int(limit), int(offset)])
        cur = self.conn.execute(sql, params)
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r["id"],
                    "uri": r["uri"],
                    "title": r["title"],
                    "size": int(r["size"]) if r["size"] is not None else 0,
                    "updated": r["updated"],
                    "score": 0.0,
                    "chunk_count": int(r["chunk_count"]) if r["chunk_count"] is not None else 0,
                    "token_count": int(r["token_count"]) if r["token_count"] is not None else 0,
                    "meta_summary": r["meta_summary"],
                }
            )
        return {"rows": rows}

    async def get_doc(self, *, id: Optional[str] = None, uri: Optional[str] = None) -> DocDetail:
        if not id and not uri:
            raise ValueError("id or uri required")
        if id:
            row = self.conn.execute(
                "SELECT doc_id, title, source, tags FROM docs WHERE doc_id=?",
                (id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT doc_id, title, source, tags FROM docs WHERE source=?",
                (uri,),
            ).fetchone()
        if not row:
            raise FileNotFoundError("document not found")
        doc_id = row["doc_id"] if "doc_id" in row.keys() else row[0]
        title = row["title"] if "title" in row.keys() else row[1]
        source = row["source"] if "source" in row.keys() else row[2]
        tags = row["tags"] if "tags" in row.keys() else row[3]
        parts = [r[0] for r in self.conn.execute(
            "SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_id",
            (doc_id,),
        ).fetchall()]
        text = "\n\n".join(p for p in parts if p)
        try:
            meta = json.loads(tags) if isinstance(tags, str) and tags else {}
        except Exception:
            meta = {}
        return {"id": doc_id, "uri": source, "title": title, "meta": meta, "text": text}

    async def export(self, *, ids: Optional[Iterable[str]] = None, q: Optional[str] = None, to_path: str) -> str:
        # Determine doc ids
        doc_ids: List[str] = []
        if ids:
            doc_ids = [str(x) for x in ids]
        elif q:
            expr = self._safe_fts_query(q)
            if expr:
                cur = self.conn.execute(
                    """
                    SELECT d.doc_id AS id, MIN(bm25(fts)) AS bm25
                    FROM fts
                    JOIN chunks c ON c.chunk_id = fts.chunk_id
                    JOIN docs d ON d.doc_id = c.doc_id
                    WHERE fts MATCH ?
                    GROUP BY d.doc_id
                    ORDER BY bm25 ASC
                    LIMIT 1000
                    """,
                    (expr,),
                )
                doc_ids = [r["id"] if "id" in r.keys() else r[0] for r in cur.fetchall()]
        else:
            cur = self.conn.execute(
                "SELECT doc_id FROM docs ORDER BY updated_at DESC LIMIT 1000"
            )
            doc_ids = [r["doc_id"] if "doc_id" in r.keys() else r[0] for r in cur.fetchall()]

        with open(to_path, "wb") as f:
            for did in doc_ids:
                row = self.conn.execute(
                    "SELECT doc_id, title, source, tags FROM docs WHERE doc_id=?",
                    (did,),
                ).fetchone()
                if not row:
                    continue
                ddoc_id = row["doc_id"] if "doc_id" in row.keys() else row[0]
                title = row["title"] if "title" in row.keys() else row[1]
                source = row["source"] if "source" in row.keys() else row[2]
                tags = row["tags"] if "tags" in row.keys() else row[3]
                parts = [r2[0] for r2 in self.conn.execute(
                    "SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_id",
                    (ddoc_id,),
                ).fetchall()]
                text = "\n\n".join(p for p in parts if p)
                try:
                    meta = json.loads(tags) if isinstance(tags, str) and tags else {}
                except Exception:
                    meta = {}
                obj = {
                    "id": ddoc_id,
                    "uri": source,
                    "title": title,
                    "meta": meta,
                    "text": text,
                }
                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                f.write(line)
        return to_path
