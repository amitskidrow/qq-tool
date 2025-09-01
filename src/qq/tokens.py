from __future__ import annotations


def rough_token_count(text: str) -> int:
    # very rough: ~4 chars per token heuristic
    return max(1, int(len(text) / 4))

