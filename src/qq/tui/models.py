from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class DocRow(TypedDict, total=False):
    id: str
    uri: Optional[str]
    title: Optional[str]
    size: int
    chunk_count: int
    token_count: int
    updated: Optional[str]
    score: float
    meta_summary: Optional[str]


class DocDetail(TypedDict, total=False):
    id: str
    uri: Optional[str]
    title: Optional[str]
    meta: Dict[str, object]
    text: str


class Page(TypedDict, total=False):
    rows: List[DocRow]

