from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import os
import numpy as np
import typesense
import httpx


TYPESENSE_HOST = os.getenv("QQ_TYPESENSE_HOST", "localhost")
try:
    TYPESENSE_PORT = int(os.getenv("QQ_TYPESENSE_PORT", "8108"))
except Exception:
    TYPESENSE_PORT = 8108
TYPESENSE_PROTOCOL = os.getenv("QQ_TYPESENSE_PROTOCOL", "http")
TYPESENSE_API_KEY = os.getenv("QQ_TYPESENSE_API_KEY", "tsdev")  # dev-only default
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
        # Reuse a persistent HTTP client for multi_search and health
        try:
            ts_timeout = float(os.getenv("QQ_TYPESENSE_TIMEOUT", "2.0"))
        except Exception:
            ts_timeout = 2.0
        self._http = httpx.Client(timeout=ts_timeout)
        self._last_stats_at: float = 0.0
        self._last_exhaustive: Optional[bool] = None

    def _multi_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a search via POST /multi_search to avoid URL length limits.

        Returns the inner single-search result object (with keys like 'hits').
        """
        payload = {
            "searches": [
                {
                    "collection": TYPESENSE_COLLECTION,
                    **params,
                }
            ]
        }
        # Prefer using the Typesense client if it exposes multi_search
        try:
            ms = getattr(self.client, "multi_search", None)
            if ms and hasattr(ms, "perform"):
                res = ms.perform(payload)
            else:
                raise AttributeError("multi_search.perform not available")
        except Exception:
            # Fallback to direct HTTP call if client-side helper is unavailable
            url = f"{TYPESENSE_PROTOCOL}://{TYPESENSE_HOST}:{TYPESENSE_PORT}/multi_search"
            headers = {
                "X-TYPESENSE-API-KEY": TYPESENSE_API_KEY,
                "Content-Type": "application/json",
            }
            r = self._http.post(url, json=payload, headers=headers)
            r.raise_for_status()
            res = r.json()

        results = res.get("results", []) if isinstance(res, dict) else []
        return results[0] if results else {"hits": []}

    # --- Schema / bootstrap ---
    def ensure_schema(self, dim: int) -> None:
        desired_dim = int(dim)
        # If collection exists, verify embedding dimension and recreate if mismatched.
        try:
            coll = self.client.collections[TYPESENSE_COLLECTION].retrieve()
            fields = {f.get("name"): f for f in coll.get("fields", [])}
            emb = fields.get("embedding") or {}
            current_dim = emb.get("num_dim")
            if current_dim is None:
                # No embedding field? Recreate collection.
                raise KeyError("embedding field missing")
            if int(current_dim) == desired_dim:
                return
            # Dimension mismatch → recreate collection with the desired dimension
            try:
                self.client.collections[TYPESENSE_COLLECTION].delete()
            except Exception:
                # If delete fails, surface original problem later
                pass
        except Exception:
            # Collection missing or malformed: proceed to create
            pass

        schema = {
            "name": TYPESENSE_COLLECTION,
            "fields": [
                {"name": "namespace", "type": "string", "facet": True},
                {"name": "source", "type": "string", "facet": True},
                {"name": "text", "type": "string"},
                {"name": "embedding", "type": "float[]", "num_dim": desired_dim},
                {"name": "created_at", "type": "int64"},
                {"name": "updated_at", "type": "int64"},
            ],
        }
        self.client.collections.create(schema)

    def _coll(self):
        return self.client.collections[TYPESENSE_COLLECTION]

    def _exhaustive(self) -> bool:
        # Cache the decision briefly to reduce control-plane calls
        now = time.time()
        if self._last_exhaustive is not None and (now - self._last_stats_at) < 2.0:
            return bool(self._last_exhaustive)
        try:
            stats = self._coll().retrieve()
            n = int(stats.get("num_documents") or 0)
            res = n <= self.flat_search_cutoff
            self._last_exhaustive = res
            self._last_stats_at = now
            return res
        except Exception:
            self._last_exhaustive = False
            self._last_stats_at = now
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
            # Stop when fewer than requested were returned (no more pages)
            if not hits or len(hits) < 250:
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
            # return only what we need; drop embedding payloads
            "include_fields": "id,source,text,namespace",
            "exclude_fields": "embedding",
        }
        if namespace:
            params["filter_by"] = f"namespace:={namespace}"
        # Use exhaustive search for tiny corpora
        if self._exhaustive():
            params["exhaustive_search"] = True
        # Use POST multi_search to avoid URL size limits when vectors are present
        res = self._multi_search(params)
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
            "include_fields": "id,source,text,namespace",
            "exclude_fields": "embedding",
        }
        if namespace:
            params["filter_by"] = f"namespace:={namespace}"
        if self._exhaustive():
            params["exhaustive_search"] = True
        # Use POST multi_search to avoid URL size limits when vectors are present
        res = self._multi_search(params)
        return self._parse_hits(res)

    def _parse_hits(self, res: Dict[str, Any]) -> List[SearchResult]:
        out: List[SearchResult] = []
        for h in res.get("hits", []):
            doc = h.get("document", {})
            # Prefer hybrid combined score when present; otherwise derive from vector distance if available;
            # finally fall back to text_match.
            score: float
            hy = h.get("hybrid_search_info", {}) if isinstance(h, dict) else {}
            comb = hy.get("combined_scores")
            if isinstance(comb, list) and comb:
                try:
                    score = float(comb[0])
                except Exception:
                    score = 0.0
            elif "vector_distance" in h:
                try:
                    vd = float(h.get("vector_distance") or 0.0)
                    # Convert smaller-is-better distance to larger-is-better similarity
                    score = 1.0 - vd
                except Exception:
                    score = 0.0
            else:
                try:
                    score = float(h.get("text_match") or 0.0)
                except Exception:
                    score = 0.0

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
            r = self._http.get(
                f"{TYPESENSE_PROTOCOL}://{TYPESENSE_HOST}:{TYPESENSE_PORT}/health",
                headers={"X-TYPESENSE-API-KEY": TYPESENSE_API_KEY},
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
