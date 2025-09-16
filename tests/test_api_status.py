import pytest

pytest.importorskip("fastapi")

import qq.qq_api as qq_api
from fastapi.testclient import TestClient


class _DummyEmbedder:
    def encode(self, items):
        return [0.0 for _ in items]


class _DummyEngine:
    def __init__(self, *_, **__):
        pass


class _DummyStore:
    def __init__(self):
        self.conn = None


def _build_app(monkeypatch, store):
    monkeypatch.setattr(qq_api, "Engine", _DummyEngine)
    monkeypatch.setattr(qq_api, "get_embedder", lambda: _DummyEmbedder())
    monkeypatch.setattr(qq_api, "Store", store)
    return qq_api.build_app()


def test_status_reports_capabilities_when_store_available(monkeypatch):
    app = _build_app(monkeypatch, _DummyStore)
    with TestClient(app) as client:
        resp = client.get("/status")
    data = resp.json()
    assert resp.status_code == 200
    assert data.get("ok") is True
    assert "index_list" in data.get("capabilities", [])
    assert isinstance(data.get("version"), str)


def test_status_omits_capabilities_when_store_missing(monkeypatch):
    app = _build_app(monkeypatch, None)
    with TestClient(app) as client:
        resp = client.get("/status")
    data = resp.json()
    assert resp.status_code == 200
    assert "index_list" not in data.get("capabilities", [])
