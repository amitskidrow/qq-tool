from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional

from .config import DEFAULT_HOME, load_config
from .util import now_ist


@dataclass
class Session:
    id: str
    created_at_ist: str
    expires_at_ist: str
    provider: str
    model: str
    system_prompt: Optional[str] = None
    token_usage: dict = None
    autoswitches: list = None
    last_ns: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _session_dir() -> Path:
    d = DEFAULT_HOME / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(session_id: str) -> Path:
    return _session_dir() / f"{session_id}.json"


def create_session(session_id: str, provider: Optional[str] = None, model: Optional[str] = None, system_prompt: Optional[str] = None) -> Session:
    cfg = load_config()
    provider = provider or cfg.get("client_model", {}).get("provider", "openai")
    model = model or cfg.get("client_model", {}).get("model", "gpt-5-mini")
    if system_prompt is None:
        system_prompt = cfg.get("client_model", {}).get("system_prompt")

    now = now_ist()
    ttl_h = int(cfg.get("sessions", {}).get("ttl_hours", 24))
    exp_dt = (now + timedelta(hours=ttl_h)) if ttl_h > 0 else now
    s = Session(
        id=session_id,
        created_at_ist=now.replace(microsecond=0).isoformat(),
        expires_at_ist=exp_dt.replace(microsecond=0).isoformat(),
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        token_usage={"input": 0, "output": 0, "total": 0},
        autoswitches=[],
        last_ns=None,
    )
    _session_path(session_id).write_text(s.to_json())
    return s


def load_session(session_id: str) -> Session:
    p = _session_path(session_id)
    data = json.loads(p.read_text())
    return Session(**data)


def save_session(s: Session) -> None:
    _session_path(s.id).write_text(s.to_json())


def list_sessions() -> list[str]:
    return [p.stem for p in _session_dir().glob("*.json")]


def close_session(session_id: str) -> None:
    p = _session_path(session_id)
    if p.exists():
        p.unlink()


def is_expired(s: Session) -> bool:
    try:
        exp = datetime.fromisoformat(s.expires_at_ist)
    except Exception:
        return False
    now = now_ist()
    # If exp is naive (shouldn't be), compare by naive local time fallback
    if exp.tzinfo is None:
        return now.replace(tzinfo=None) >= exp
    return now >= exp
