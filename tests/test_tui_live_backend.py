import pytest

pytest.importorskip("textual")

import httpx

from qq.tui.backends.live import LiveBackend


class _StubClient:
    async def get(self, *_, **__):
        return httpx.Response(404, request=httpx.Request("GET", "http://qq.local/index/list"))

    async def stream(self, *_, **__):
        response = httpx.Response(404, request=httpx.Request("GET", "http://qq.local/export"))

        class _Stream:
            async def __aenter__(self_nonlocal):
                return response

            async def __aexit__(self_nonlocal, exc_type, exc, tb):
                return False

        return _Stream()


@pytest.mark.asyncio
async def test_live_backend_raises_clear_error_on_missing_route(monkeypatch):
    backend = LiveBackend()

    async def _fake_client_get():
        return _StubClient()

    monkeypatch.setattr(backend, "_client_get", _fake_client_get)

    with pytest.raises(RuntimeError) as excinfo:
        await backend.list_docs(None, None, 10, 0)
    assert "does not expose" in str(excinfo.value)
