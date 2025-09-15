from __future__ import annotations

import os
from typing import Optional


def uds_path() -> str:
    return os.getenv("QQ_UDS", "/run/qq.sock")


def snapshot_path(explicit: Optional[str]) -> Optional[str]:
    return explicit

