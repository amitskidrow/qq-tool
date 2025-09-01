from __future__ import annotations

from typing import Dict, List, Optional

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct, PointIdsList
import uuid

from ..audit import append_audit
from ..embeddings import embed_texts
from ..hashing import content_hash
from ..policy import guard_write
from ..simhash import simhash_text64
from ..util import now_ist


def kb_search(client, collection: str, ns: str, q: str, topk: int = 5):
    from ..embeddings import embed_texts

    qv = embed_texts([q])[0]
    filt = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=ns))])
    res = client.search(collection_name=collection, query_vector=qv.tolist(), limit=topk, query_filter=filt)
    out = []
    for p in res:
        pl = p.payload or {}
        out.append({"id": p.id, "score": float(p.score or 0.0), "text": pl.get("text"), "source": pl.get("source")})
    return out


def kb_insert(client, collection: str, payload: Dict, interactive: bool = False, force: bool = False):
    ok, err = guard_write(payload, interactive, force)
    if not ok:
        raise RuntimeError(err)
    text = payload.get("text") or ""
    v = embed_texts([text])[0]
    h = content_hash(text)
    sh = int(simhash_text64(text))
    # Deterministic UUID to satisfy Qdrant's point ID requirements
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{h}-0"))
    payload = {**payload, "hash": h, "simhash": sh}
    pt = PointStruct(id=pid, vector=v.tolist(), payload=payload)
    client.upsert(collection_name=collection, points=[pt], wait=True)
    append_audit({"ts": now_ist().isoformat(), "action": "insert", "id": pid, "payload": payload})
    return pid


def kb_update(client, collection: str, point_id: str, payload: Dict, interactive: bool = False, force: bool = False):
    ok, err = guard_write(payload, interactive, force)
    if not ok:
        raise RuntimeError(err)
    # simple overwrite of payload fields
    client.set_payload(collection_name=collection, payload=payload, points=[point_id])
    append_audit({"ts": now_ist().isoformat(), "action": "update", "id": point_id, "payload": payload})
    return point_id


def kb_delete(client, collection: str, ids: List[str]):
    client.delete(collection_name=collection, points_selector=PointIdsList(points=ids))
    append_audit({"ts": now_ist().isoformat(), "action": "delete", "ids": ids})
    return ids
