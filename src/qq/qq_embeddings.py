from __future__ import annotations

import os
import time
from dataclasses import dataclass
import warnings
from functools import lru_cache
from typing import Iterable, List

import numpy as np


@dataclass
class EmbedInfo:
    model_name: str
    backend: str
    dim: int
    warmed: bool
    load_ms: float


class OnnxEmbedder:
    """
    Sentence embeddings via ONNX Runtime using all-MiniLM-L6-v2 (384-dim).

    Uses HuggingFace transformers + optimum.onnxruntime to export/load
    an ONNX feature-extraction model, then performs mean pooling and L2
    normalization to match Sentence-Transformers behavior.
    """

    def __init__(self, model_name: str | None = None):
        t0 = time.perf_counter()
        self.model_name = model_name or os.getenv(
            "QQ_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Keep HF cache in RAM if desired
        hf_home = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        if hf_home:
            os.makedirs(hf_home, exist_ok=True)

        # Lazy imports to avoid heavy import time when not used
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Export to ONNX at first load if needed
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_name, export=True
        )
        # Determine output dimension by probing
        vec = self.encode(["dim-probe"])
        self._dim = int(vec.shape[1])
        self._load_ms = (time.perf_counter() - t0) * 1000.0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def info(self) -> EmbedInfo:
        return EmbedInfo(
            model_name=self.model_name,
            backend="onnxruntime",
            dim=self._dim,
            warmed=True,
            load_ms=self._load_ms,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        from numpy.linalg import norm

        batch = list(texts)
        if not batch:
            return np.zeros((0, self._dim), dtype=np.float32)

        inputs = self.tokenizer(
            batch, padding=True, truncation=True, return_tensors="np"
        )
        # Run ONNX model
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        mask = inputs["attention_mask"][..., None]  # [B, T, 1]
        # Mean pool
        summed = (last_hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        # L2 normalize
        n = norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(n, 1e-9, None)
        return pooled.astype(np.float32)


class SimpleEmbedder:
    """
    Lightweight, dependency-free fallback embedder for local dev/testing.
    Produces deterministic 384-dim bag-of-words hashed embeddings with L2 norm.
    Activate with env QQ_EMBED_STUB=1 to skip model downloads.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.model_name = "stub-hash-embedder"
        self._load_ms = 0.0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def info(self) -> EmbedInfo:
        return EmbedInfo(
            model_name=self.model_name,
            backend="python-stub",
            dim=self._dim,
            warmed=True,
            load_ms=self._load_ms,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        from numpy.linalg import norm
        arr: list[list[float]] = []
        for t in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for tok in str(t).lower().split():
                h = int.from_bytes(
                    np.frombuffer(
                        bytes(tok.encode("utf-8")), dtype=np.uint8
                    ).tobytes(),
                    "little",
                    signed=False,
                )
                idx = h % self._dim
                sign = 1.0 if (h >> 1) & 1 else -1.0
                vec[idx] += sign
            n = norm(vec) or 1.0
            arr.append((vec / n).astype(np.float32))
        if not arr:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack(arr, axis=0)


@lru_cache(maxsize=1)
def get_embedder():
    use_stub = os.getenv("QQ_EMBED_STUB", "0").lower() in {"1", "true", "yes"}
    if use_stub:
        warnings.warn("QQ_EMBED_STUB=1 set; using SimpleEmbedder for testing.")
        return SimpleEmbedder()
    try:
        return OnnxEmbedder()
    except Exception as e:  # pragma: no cover
        warnings.warn(
            f"Falling back to SimpleEmbedder due to OnnxEmbedder init error: {e}"
        )
        return SimpleEmbedder()
