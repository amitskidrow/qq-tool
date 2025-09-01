from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_HOME
from .util import now_ist


def append_audit(event: Dict[str, Any]) -> None:
    d = DEFAULT_HOME / "audit"
    d.mkdir(parents=True, exist_ok=True)
    date_str = now_ist().strftime("%Y%m%d")
    p = d / f"{date_str}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

