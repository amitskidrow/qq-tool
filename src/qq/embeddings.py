from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings."
        ) from e
    # Allow overriding the embedding model for latency/quality tradeoffs
    # Examples: "all-MiniLM-L6-v2" (~384d, very fast), "BAAI/bge-small-en-v1.5"
    model_name = os.getenv("QQ_EMBED_MODEL", "BAAI/bge-m3")
    model = SentenceTransformer(model_name)
    return model


def embedding_dim() -> int:
    model = _get_model()
    v = model.encode(["dim-probe"], normalize_embeddings=True)
    return int(v.shape[1])


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return vecs.astype(np.float32)
