import sys
from typing import List, Optional

from . import __version__
try:
    # New UDS-based CLI
    from .qq_cli import app  # type: ignore
except Exception:
    # Fallback to legacy CLI if import fails
    from .cli import app  # type: ignore


def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    # Handle common version flags without invoking Typer parsing
    if argv and argv[0] in {"--version", "-V"}:
        print(__version__)
        return

    # Fallback to Typer app (supports 'version' command too)
    app()
