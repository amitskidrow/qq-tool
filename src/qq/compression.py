from __future__ import annotations

from typing import List, Tuple, Optional

from .tokens import rough_token_count


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    # Very rough: assume ~4 chars/token heuristic, but keep deterministic substring
    approx_chars = max_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[:approx_chars]


def compress_texts(texts: List[str], *, ratio: Optional[float] = None, max_tokens: Optional[int] = None) -> Tuple[List[str], float]:
    """Deterministic extractive compression.

    - If ratio is provided (0<ratio<=1), trims each text proportionally by tokens.
    - If max_tokens is provided, trims the concatenated output to the budget and then maps
      back to per-text slices in original order.
    Returns (compressed_texts, applied_ratio).
    """
    if not texts:
        return [], 1.0

    if max_tokens is not None:
        # Global budget: slice texts in order until budget is exhausted
        remaining = max(0, int(max_tokens))
        out: List[str] = []
        total_tokens = sum(rough_token_count(t) for t in texts) or 1
        applied_ratio = min(1.0, remaining / total_tokens)
        for t in texts:
            if remaining <= 0:
                out.append("")
                continue
            need = min(rough_token_count(t), remaining)
            out.append(_truncate_to_tokens(t, need))
            remaining -= need
        return out, applied_ratio

    if ratio is None:
        ratio = 1.0
    ratio = max(0.01, min(1.0, float(ratio)))
    out: List[str] = []
    for t in texts:
        tk = max(1, int(rough_token_count(t) * ratio))
        out.append(_truncate_to_tokens(t, tk))
    return out, ratio

