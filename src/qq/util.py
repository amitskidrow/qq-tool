from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def to_slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "project"


def git_project_name(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd or "."),
            capture_output=True,
            text=True,
            check=True,
        )
        root = Path(proc.stdout.strip())
        return root.name
    except Exception:
        return None


def infer_namespace(cwd: Optional[Path] = None) -> str:
    name = git_project_name(cwd)
    if name:
        return f"project:{to_slug(name)}"
    return "global"


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def is_supported_file(path: Path) -> bool:
    exts = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".sh", ".toml"}
    return path.is_file() and path.suffix.lower() in exts

