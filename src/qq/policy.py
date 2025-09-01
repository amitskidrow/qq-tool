from __future__ import annotations

import json
from typing import Dict, Optional

import httpx

from .config import load_config


def opa_decide(input_doc: Dict) -> Optional[Dict]:
    cfg = load_config().get("policy", {}).get("opa", {})
    if not cfg.get("enabled", True):
        return None
    url = cfg.get("url") or "http://localhost:8181/v1/data/qq/allow"
    try:
        r = httpx.post(url, json={"input": input_doc}, timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def guard_write(payload: Dict, interactive: bool, force: bool = False) -> tuple[bool, str | None]:
    # local severity gate
    t = payload.get("type")
    severity = int(payload.get("severity", 0) or 0)
    if t == "global_approach" and severity >= 2 and not interactive and not force:
        return False, "ERR_POLICY_GUARD"
    # OPA hook
    decision = opa_decide({"action": "write", "payload": payload, "interactive": interactive, "force": force})
    if decision is not None:
        allow = bool(decision.get("result", {}).get("allow", True))
        if not allow:
            return False, "ERR_POLICY_GUARD"
    return True, None

