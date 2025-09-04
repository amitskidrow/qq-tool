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
    # Fast default: MiniLM-L6-v2 (384-dim)
    # Override via QQ_EMBED_MODEL if needed.
    model_name = os.getenv("QQ_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device = os.getenv("QQ_EMBED_DEVICE")
    try:
        if device:
            model = SentenceTransformer(model_name, device=device)
        else:
            model = SentenceTransformer(model_name)
    except Exception:
        # Fallback to CPU if requested device isn't available
        model = SentenceTransformer(model_name, device="cpu")
    return model


def embedding_dim() -> int:
    model = _get_model()
    v = model.encode(["dim-probe"], normalize_embeddings=True)
    return int(v.shape[1])


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = _get_model()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return vecs.astype(np.float32)
