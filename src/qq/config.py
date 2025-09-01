from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_DIR_ENV = "QQ_HOME"
# Only use env if set and non-empty; otherwise fallback to $HOME/.qq
_env_home = os.environ.get(CONFIG_DIR_ENV)
DEFAULT_HOME = Path(_env_home) if _env_home else (Path.home() / ".qq")
CONFIG_PATH = DEFAULT_HOME / "config.yaml"


def ensure_dirs() -> None:
    DEFAULT_HOME.mkdir(parents=True, exist_ok=True)
    (DEFAULT_HOME / "sessions").mkdir(parents=True, exist_ok=True)
    (DEFAULT_HOME / "audit").mkdir(parents=True, exist_ok=True)
    (DEFAULT_HOME / "todos").mkdir(parents=True, exist_ok=True)
    (DEFAULT_HOME / "usage").mkdir(parents=True, exist_ok=True)


def default_config() -> Dict[str, Any]:
    return {
        "client_model": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "system_prompt": None,
            "token_caps": {
                "gpt-5": 250_000,
                "gpt-5-mini": 2_000_000,
                "gpt-5-nano": 2_000_000,
            },
            "autoswitch": "gemini-2.5-pro",
        },
        "sessions": {
            "timezone": "Asia/Kolkata",
            "ttl_hours": 24,
        },
        "database": {
            # Local SQLite database path for retrieval store
            "path": str(DEFAULT_HOME / "qq.db"),
        },
        "usage": {
            "mode": "daily",
            "timezone": "Asia/Kolkata",
        },
        "retrieval": {
            "reranker": {"enabled": False, "model": "BAAI/bge-reranker-large"},
        },
        "policy": {
            "opa": {"enabled": True, "url": None},  # use local OPA if available
        },
    }


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    return default_config()


def save_config(cfg: Dict[str, Any]) -> None:
    ensure_dirs()
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
