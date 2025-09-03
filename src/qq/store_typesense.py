from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import typesense
import httpx


TYPESENSE_HOST = "localhost"
TYPESENSE_PORT = 8108
TYPESENSE_PROTOCOL = "http"
TYPESENSE_API_KEY = "tsdev"  # dev-only, embedded on purpose
TYPESENSE_COLLECTION = "qq_docs"


def _to_unix(ts: Optional[str]) -> int:
    if not ts:
        return int(time.time())
    try:
        # Expect ISO format
        return int(datetime.fromisoformat(ts).timestamp())
    except Exception:
        return int(time.time())


@dataclass
class SearchResult:
    id: str
    score: float
    uri: str
    namespace: Optional[str]
    title: Optional[str]
    meta: Dict[str, Any]
    text: str


class TypesenseStore:
    def __init__(self, *, flat_search_cutoff: int = 20):
        self.client = typesense.Client(
            {
                "nodes": [
                    {
                        "host": TYPESENSE_HOST,
                        "port": TYPESENSE_PORT,
                        "protocol": TYPESENSE_PROTOCOL,
                    }
                ],
                "api_key": TYPESENSE_API_KEY,
                "connection_timeout_seconds": 2,
            }
        )
        self.flat_search_cutoff = int(max(0, flat_search_cutoff))

    # --- Schema / bootstrap ---
    def ensure_schema(self, dim: int) -> None:
        try:
            self.client.collections[TYPESENSE_COLLECTION].retrieve()
            return
        except Exception:
            pass

        schema = {
            "name": TYPESENSE_COLLECTION,
            "fields": [
                {"name": "namespace", "type": "string", "facet": True},
                {"name": "source", "type": "string", "facet": True},
                {"name": "text", "type": "string"},
                {"name": "embedding", "type": "float[]", "num_dim": int(dim)},
                {"name": "created_at", "type": "int64"},
                {"name": "updated_at", "type": "int64"},
            ],
        }
        self.client.collections.create(schema)

    def _coll(self):
        return self.client.collections[TYPESENSE_COLLECTION]

    def _exhaustive(self) -> bool:
        try:
            stats = self._coll().retrieve()
            n = int(stats.get("num_documents") or 0)
            return n <= self.flat_search_cutoff
        except Exception:
            return False

    # --- CRUD ---
    def upsert_chunk(
        self,
        *,
        uri: str,
        namespace: str,
        text: str,
        vector: np.ndarray,
        title: Optional[str] = None,  # ignored, kept for compatibility
        meta: Optional[Dict[str, Any]] = None,  # ignored
        hash_: Optional[str] = None,
        simhash: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> str:
        # Stable id based on primary fields, to allow idempotent upserts
        h = hashlib.sha1()
        h.update((uri + "|" + namespace + "|" + (hash_ or "") + "|" + (title or "") + "|" + text).encode("utf-8"))
        doc_id = h.hexdigest()
        doc = {
            "id": doc_id,
            "namespace": namespace,
            "source": uri,
            "text": text,
            "embedding": vector.tolist(),
            "created_at": _to_unix(created_at),
            "updated_at": _to_unix(updated_at),
        }
        self._coll().documents.upsert(doc)
        return doc_id

    def delete_ids(self, ids: Sequence[str]) -> int:
        deleted = 0
        for i in ids:
            try:
                self._coll().documents[i].delete()
                deleted += 1
            except Exception:
                pass
        return deleted

    def list_by_uri(self, uri: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        page = 1
        while True:
            res = self._coll().documents.search(
                {
                    "q": "*",
                    "query_by": "text",
                    "filter_by": f"source:={uri}",
                    "per_page": 250,
                    "page": page,
                }
            )
            hits = res.get("hits", [])
            for h in hits:
                doc = h.get("document", {})
                out.append(doc)
            if page >= int(res.get("out_of", 0)):
                break
            page += 1
        return out

    def list_namespace(self, namespace: str, limit: int = 1000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        page = 1
        remain = max(1, limit)
        while remain > 0:
            per_page = min(250, remain)
            res = self._coll().documents.search(
                {
                    "q": "*",
                    "query_by": "text",
                    "filter_by": f"namespace:={namespace}",
                    "per_page": per_page,
                    "page": page,
                }
            )
            hits = res.get("hits", [])
            for h in hits:
                out.append(h.get("document", {}))
            remain -= len(hits)
            if not hits or len(hits) < per_page:
                break
            page += 1
        return out

    # --- Search ---
    def search_dense(
        self, query_vec: np.ndarray, namespace: Optional[str] = None, topk: int = 8
    ) -> List[SearchResult]:
        vec = ",".join(str(float(x)) for x in query_vec.tolist())
        params: Dict[str, Any] = {
            "q": "*",
            "query_by": "text",
            "vector_query": f"embedding:([{vec}], k:{int(max(1, topk))})",
            "per_page": int(max(1, topk)),
        }
        if namespace:
            params["filter_by"] = f"namespace:={namespace}"
        # Use exhaustive search for tiny corpora
        if self._exhaustive():
            params["exhaustive_search"] = True
        res = self._coll().documents.search(params)
        return self._parse_hits(res)

    def search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text: str,
        namespace: Optional[str] = None,
        topk: int = 8,
        alpha: float = 0.55,
        pool: int = 200,
    ) -> List[SearchResult]:
        vec = ",".join(str(float(x)) for x in query_vec.tolist())
        params: Dict[str, Any] = {
            "q": query_text,
            "query_by": "text",
            "vector_query": f"embedding:([{vec}], k:{int(max(1, topk))}, alpha:{max(0.0, min(1.0, float(alpha))):.2f})",
            "per_page": int(max(1, topk)),
        }
        if namespace:
            params["filter_by"] = f"namespace:={namespace}"
        if self._exhaustive():
            params["exhaustive_search"] = True
        res = self._coll().documents.search(params)
        return self._parse_hits(res)

    def _parse_hits(self, res: Dict[str, Any]) -> List[SearchResult]:
        out: List[SearchResult] = []
        for h in res.get("hits", []):
            doc = h.get("document", {})
            score = float(h.get("hybrid_search_info", {}).get("combined_scores", [h.get("text_match", 0.0)])[0] if isinstance(h.get("hybrid_search_info", {}).get("combined_scores"), list) else h.get("text_match", 0.0))
            out.append(
                SearchResult(
                    id=str(doc.get("id")),
                    score=score,
                    uri=str(doc.get("source")),
                    namespace=str(doc.get("namespace")) if doc.get("namespace") is not None else None,
                    title=None,
                    meta={},
                    text=str(doc.get("text")) if doc.get("text") is not None else "",
                )
            )
        return out

    # --- Health / doctor ---
    def doctor(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": True, "checks": []}

        def check(name: str, ok: bool, detail: Optional[str] = None):
            out.setdefault("checks", []).append({"name": name, "ok": ok, "detail": detail})
            if not ok:
                out["ok"] = False

        # HTTP health
        try:
            r = httpx.get(
                f"{TYPESENSE_PROTOCOL}://{TYPESENSE_HOST}:{TYPESENSE_PORT}/health",
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
                timeout=2.0,
            )
            check("http_health", r.status_code == 200, f"status={r.status_code}")
        except Exception as e:
            check("http_health", False, str(e))

        # Collection + dimension
        try:
            c = self._coll().retrieve()
            fields = {f.get("name"): f for f in c.get("fields", [])}
            emb = fields.get("embedding") or {}
            dim = emb.get("num_dim")
            ok = dim is not None
            check("collection_present", True, f"name={c.get('name')}, dim={dim}")
            check("dimension_set", ok, f"dim={dim}")
        except Exception as e:
            check("collection_present", False, str(e))
            check("dimension_set", False, None)

        return out

