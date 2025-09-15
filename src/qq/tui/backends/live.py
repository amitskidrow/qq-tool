from __future__ import annotations

from typing import Iterable, Optional

import httpx

from ..config import uds_path
from ..models import DocDetail, Page
from .base import Backend


class LiveBackend(Backend):
    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None:
            try:
                transport = httpx.AsyncHTTPTransport(uds=uds_path())
                self._client = httpx.AsyncClient(
                    transport=transport,
                    base_url="http://qq.local",
                    timeout=30.0,
                )
            except AttributeError as e:
                # Friendly message for transport mismatch on older code/installs
                raise RuntimeError(
                    "httpx transport mismatch: AsyncClient requires AsyncHTTPTransport. "
                    "Upgrade qq/httpx and try again (pip install -U httpx)."
                ) from e
            except OSError as e:
                # Common UDS errors: not found or permission denied
                import errno as _errno

                if e.errno == _errno.ENOENT:
                    raise RuntimeError(
                        f"UDS not found at {uds_path()}. Ensure the qq API is running and QQ_UDS is correct."
                    ) from e
                if e.errno in (_errno.EPERM, _errno.EACCES):
                    raise RuntimeError(
                        f"Permission denied opening UDS at {uds_path()}. Try sudo or adjust socket perms."
                    ) from e
                raise
        return self._client

    async def list_docs(self, q: Optional[str], like: Optional[str], limit: int, offset: int) -> Page:
        c = await self._client_get()
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        elif like:
            params["like"] = like
        r = await c.get("/index/list", params=params)
        r.raise_for_status()
        return r.json()

    async def get_doc(self, *, id: Optional[str] = None, uri: Optional[str] = None) -> DocDetail:
        if not id and not uri:
            raise ValueError("id or uri required")
        c = await self._client_get()
        params = {}
        if id:
            params["id"] = id
        if uri:
            params["uri"] = uri
        r = await c.get("/index/get", params=params)
        r.raise_for_status()
        return r.json()

    async def export(self, *, ids: Optional[Iterable[str]] = None, q: Optional[str] = None, to_path: str) -> str:
        c = await self._client_get()
        params = {}
        if ids:
            params["ids"] = ",".join(ids)
        elif q:
            params["q"] = q
        async with c.stream("GET", "/export", params=params) as resp:
            resp.raise_for_status()
            with open(to_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
        return to_path
