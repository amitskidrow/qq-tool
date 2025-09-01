from __future__ import annotations

from typing import List, Tuple

from rank_bm25 import BM25Okapi


def bm25_rank(query: str, docs: List[str]) -> List[Tuple[int, float]]:
    # Tokenize simply by whitespace; can be improved
    corpus = [d.split() for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.split())
    idx_scores = list(enumerate(map(float, scores)))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    return idx_scores


def hybrid_merge(dense: List[Tuple[int, float]], sparse: List[Tuple[int, float]], alpha: float = 0.5) -> List[Tuple[int, float]]:
    # dense and sparse are lists of (index, score). Normalize scores to [0,1] by rank.
    def normalize(ranked: List[Tuple[int, float]]):
        if not ranked:
            return {}
        n = len(ranked)
        # convert rank -> 1.0 for top, 0.0 for last
        return {idx: 1.0 - (i / max(n - 1, 1)) for i, (idx, _s) in enumerate(ranked)}

    nd = normalize(dense)
    ns = normalize(sparse)
    keys = set(nd) | set(ns)
    merged = []
    for k in keys:
        sc = (alpha * nd.get(k, 0.0)) + ((1 - alpha) * ns.get(k, 0.0))
        merged.append((k, sc))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

