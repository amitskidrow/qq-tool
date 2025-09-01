from __future__ import annotations

from typing import List


def simple_chunks(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    # naive paragraph/sentence split with sliding window
    # split by double newlines first
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: list[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur = cur + "\n\n" + p
        else:
            parts.append(cur)
            cur = p
    if cur:
        parts.append(cur)
    # re-window with overlap
    out: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        out.append(chunk)
        if overlap and i + 1 < len(parts):
            # create synthetic overlap by prepending tail of current
            tail = chunk[-overlap:]
            parts[i + 1] = tail + "\n\n" + parts[i + 1]
        i += 1
    return out

