from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .config import DEFAULT_HOME, load_config
from .util import now_ist


def _usage_path() -> Path:
    d = DEFAULT_HOME / "usage"
    d.mkdir(parents=True, exist_ok=True)
    day = now_ist().strftime("%Y%m%d")
    return d / f"{day}.json"


def _load() -> Dict[str, Dict[str, int]]:
    p = _usage_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save(data: Dict[str, Dict[str, int]]) -> None:
    p = _usage_path()
    p.write_text(json.dumps(data))


def get_caps() -> Dict[str, int]:
    cfg = load_config()
    return cfg.get("client_model", {}).get("token_caps", {})


def add_usage(provider: str, model: str, tokens: int) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    data = _load()
    prov = data.setdefault(provider, {})
    current = int(prov.get(model, 0) or 0)
    prov[model] = current + int(tokens)
    _save(data)
    caps = get_caps()
    cap = caps.get(model)
    ok = True if cap is None else prov[model] <= cap
    return ok, data


def get_usage() -> Dict[str, Dict[str, int]]:
    return _load()


def remaining(model: str) -> int | None:
    caps = get_caps()
    use = _load()
    total = 0
    for prov in use.values():
        total += int(prov.get(model, 0) or 0)
    if model not in caps:
        return None
    return max(caps[model] - total, 0)

