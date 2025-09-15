from __future__ import annotations

import abc
from typing import Iterable, List, Optional

from ..models import DocDetail, DocRow, Page


class Backend(abc.ABC):
    @abc.abstractmethod
    async def list_docs(self, q: Optional[str], like: Optional[str], limit: int, offset: int) -> Page:  # noqa: D401
        """List documents (paged)."""

    @abc.abstractmethod
    async def get_doc(self, *, id: Optional[str] = None, uri: Optional[str] = None) -> DocDetail:  # noqa: D401
        """Fetch a full document by id or uri."""

    @abc.abstractmethod
    async def export(self, *, ids: Optional[Iterable[str]] = None, q: Optional[str] = None, to_path: str) -> str:  # noqa: D401
        """Export current selection to a JSONL file and return path."""
