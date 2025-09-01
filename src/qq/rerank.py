from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple


@lru_cache(maxsize=1)
def _get_ce(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for reranking") from e
    return CrossEncoder(model_name)


def rerank_pairs(model_name: str, query: str, passages: List[str]) -> List[Tuple[int, float]]:
    ce = _get_ce(model_name)
    pairs = [(query, p) for p in passages]
    scores = ce.predict(pairs)
    # return indices sorted by score desc
    idx_scores = list(enumerate(map(float, scores)))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    return idx_scores

