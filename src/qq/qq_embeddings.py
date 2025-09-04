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
    providers: List[str] | None = None


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
        import onnxruntime as ort

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Decide ONNX Runtime execution provider(s)
        # - Default to CPU; allow opting into ROCm (AMD) or CUDA via env
        # - Env: QQ_ORT_PROVIDER (e.g., "ROCMExecutionProvider") or
        #        QQ_ORT_PROVIDERS (comma separated list)
        env_single = (os.getenv("QQ_ORT_PROVIDER") or "").strip()
        env_list = (os.getenv("QQ_ORT_PROVIDERS") or "").strip()

        def _normalize_provider(name: str) -> str:
            n = name.strip().lower()
            if n in {"cpu", "cpuep", "cpuexecutionprovider"}:
                return "CPUExecutionProvider"
            if n in {"cuda", "cudaep", "cudaexecutionprovider"}:
                return "CUDAExecutionProvider"
            if n in {"rocm", "rocmep", "rocmexecutionprovider"}:  # permissive
                return "ROCMExecutionProvider"
            if n in {"dml", "dmlep", "dmlexecutionprovider"}:
                return "DmlExecutionProvider"
            if n in {"openvino", "openvinoexecutionprovider"}:
                return "OpenVINOExecutionProvider"
            return name  # assume exact provider name

        providers: list[str]
        if env_list:
            providers = [_normalize_provider(p) for p in env_list.split(",") if p.strip()]
        elif env_single:
            providers = [_normalize_provider(env_single)]
        else:
            # Auto: prefer ROCm on AMD if available, else CPU
            avail = set(ort.get_available_providers())
            if "ROCMExecutionProvider" in avail:
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        # Export to ONNX at first load if needed, requesting chosen provider(s) when supported
        model_loaded = False
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                # Try new API with providers kw
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    self.model_name, export=True, providers=providers
                )
                model_loaded = True
                break
            except TypeError as e:
                # Older optimum versions may not accept providers=; try provider=
                last_err = e
                try:
                    self.model = ORTModelForFeatureExtraction.from_pretrained(
                        self.model_name, export=True, provider=providers[0]
                    )
                    model_loaded = True
                    break
                except TypeError as e2:
                    last_err = e2
                    # Fall through to plain call without provider specification
            except Exception as e:
                # Provider not available or other runtime error; fall back to CPU once
                last_err = e
                if attempt == 0 and providers and providers[0] != "CPUExecutionProvider":
                    providers = ["CPUExecutionProvider"]
                    continue
                break

        if not model_loaded:
            # Final fallback without explicit provider selection
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name, export=True
            )
            if last_err is not None:
                warnings.warn(
                    f"ORT provider selection fallback to CPU due to: {last_err}"
                )

        # Remember chosen providers (best-effort)
        try:
            self.providers = list(getattr(self.model, "providers", providers))  # type: ignore[attr-defined]
        except Exception:
            self.providers = providers
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
            providers=list(getattr(self, "providers", []) or []),
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
            providers=None,
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
