from __future__ import annotations

import hashlib
import re
from typing import Iterable


def _tokenize(text: str) -> list[str]:
    # simple word tokens; can be improved with n-grams
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _hash64(s: str) -> int:
    # 64-bit hash from sha1
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:16], 16)


def simhash64(tokens: Iterable[str]) -> int:
    v = [0] * 64
    for t in tokens:
        h = _hash64(t)
        for i in range(64):
            bit = 1 if (h >> i) & 1 else 0
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def simhash_text64(text: str) -> int:
    return simhash64(_tokenize(text))


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

