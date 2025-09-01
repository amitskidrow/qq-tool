from __future__ import annotations

from typing import Optional
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def connect(url: str, api_key: Optional[str] = None) -> QdrantClient:
    # Allow env override when not provided via config
    if api_key is None:
        api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, name: str, vector_size: int, recreate: bool = False) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing and not recreate:
        return
    if name in existing and recreate:
        client.delete_collection(collection_name=name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
