from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

from .embeddings import embed_texts, embedding_dim
try:
    # Best-effort introspection of model/device
    from .embeddings import _get_model  # type: ignore
except Exception:  # pragma: no cover
    _get_model = None  # type: ignore
from .store_typesense import TypesenseStore
from .util import infer_namespace


def build_app() -> FastAPI:
    app = FastAPI(title="qq API", version="0.0.1", default_response_class=ORJSONResponse)
    app.state.embed_info = {
        "model": None,
        "device": None,
        "dim": None,
        "cuda": None,
        "warmed": False,
    }

    # Warm the embedding model and prepare Typesense on startup to avoid first-request latency
    @app.on_event("startup")
    def _warmup():
        try:
            # Load model into memory and infer embedding dim
            _ = embed_texts(["warmup"])
            dim = embedding_dim()
        except Exception:
            dim = None

        # Reuse a single Typesense client across requests
        try:
            app.state.store = TypesenseStore()
            if dim is not None:
                app.state.store.ensure_schema(dim)
        except Exception:
            # Defer failures to request-time exceptions
            app.state.store = TypesenseStore()

        # Record diagnostic info about the embedder and device
        try:
            info = app.state.embed_info
            if _get_model is not None:
                m = _get_model()
                info["model"] = getattr(m, "__class__", type(m)).__name__
                # SentenceTransformer exposes .device on the underlying ._first_module().
                device = None
                try:
                    device = str(getattr(m, "device", None) or getattr(m, "_target_device", None))
                except Exception:
                    pass
                info["device"] = device
            # Torch device status (optional)
            try:  # pragma: no cover
                import torch

                info["cuda"] = bool(torch.cuda.is_available())
                try:
                    info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                except Exception:
                    info["cuda_device_name"] = None
            except Exception:
                info["cuda"] = None
                info["cuda_device_name"] = None
            info["dim"] = int(dim) if dim is not None else None
            info["warmed"] = True
        except Exception:
            pass

    @app.get("/health")
    def health():
        # Include diagnostics to quickly verify the runtime device/model
        info = getattr(app.state, "embed_info", {})
        return {"ok": True, "embed": info}

    # Alias for readiness probes
    @app.get("/ready")
    def ready():
        info = getattr(app.state, "embed_info", {})
        return {"ok": True, "embed": info}

    @app.get("/query")
    def api_query(q: str, ns: str | None = None, topk: int = 5, hybrid: bool = True):
        store: TypesenseStore = getattr(app.state, "store", TypesenseStore())
        try:
            q_vec = embed_texts([q])[0]
            ns_final = ns or infer_namespace()
            if hybrid:
                results = store.search_hybrid(q_vec, q, namespace=ns_final, topk=topk)
            else:
                results = store.search_dense(q_vec, namespace=ns_final, topk=topk)
            out = []
            for r in results:
                out.append({
                    "id": r.id,
                    "score": r.score,
                    "source": r.uri,
                    "namespace": r.namespace,
                    "snippet": (r.text or "")[:240],
                })
            return {"results": out, "ns": ns_final}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

# Export module-level app for Uvicorn discovery (uvicorn qq.api:app)
app = build_app()
