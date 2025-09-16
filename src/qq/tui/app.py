from __future__ import annotations

import asyncio
import datetime as _dt
from pathlib import Path
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable, Input, Static

from .backends.base import Backend
from .backends.live import LiveBackend
from .backends.snapshot import SnapshotBackend
from .models import DocDetail, DocRow


class Preview(Static):
    def set_content(self, text: str, meta: dict | None = None) -> None:
        parts: List[str] = []
        if meta:
            try:
                kv = ", ".join(f"{k}={v}" for k, v in meta.items())
                parts.append(f"meta: {kv}")
            except Exception:
                pass
        if text:
            parts.append(text)
        self.update("\n\n".join(parts))


class QQTui(App):
    CSS = ""
    BINDINGS = [
        Binding("/", "focus_search", "Search"),
        Binding("enter", "open_row", "Open"),
        Binding("e", "export_rows", "Export"),
        Binding("b", "toggle_backend", "Backend"),
        Binding("q", "quit", "Quit"),
    ]

    q: reactive[str | None] = reactive(None)
    like: reactive[str | None] = reactive(None)
    limit: int = 50
    offset: reactive[int] = reactive(0)

    def __init__(self, backend: Backend, alt_backend: Backend | None = None) -> None:
        super().__init__()
        self.backend = backend
        self.alt_backend = alt_backend
        self.rows: List[DocRow] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Input(placeholder="Search (/ to focus; Enter to submit)", id="search")
            with Horizontal():
                yield DataTable(id="table")
                yield Preview(id="preview")

    async def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("id", "uri", "title", "size", "score", "chunks", "tokens")
        table.cursor_type = "row"
        await self._load_page(reset=True)

    async def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        self.q = text or None
        self.offset = 0
        await self._load_page(reset=True)

    async def action_open_row(self) -> None:
        table = self.query_one(DataTable)
        if table.row_count == 0 or table.cursor_row is None:
            return
        try:
            row = self.rows[table.cursor_row]
        except Exception:
            return
        await self._load_detail(row)

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:  # type: ignore[override]
        # Lazy-load detail on highlight for responsiveness
        try:
            row = self.rows[event.row_index]
        except Exception:
            return
        await self._load_detail(row, preview_only=True)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:  # type: ignore[override]
        try:
            row = self.rows[event.row_index]
        except Exception:
            return
        await self._load_detail(row)

    async def _load_detail(self, row: DocRow, preview_only: bool = False) -> None:
        try:
            detail: DocDetail = await self.backend.get_doc(id=row.get("id"))
            text = detail.get("text") or ""
            if preview_only and len(text) > 2000:
                text = text[:2000] + "\n\n… (truncated)"
            self.query_one(Preview).set_content(text, meta=detail.get("meta") or {})
        except Exception as e:
            self.query_one(Preview).update(f"error: {e}")

    async def _load_page(self, reset: bool = False) -> None:
        try:
            page = await self.backend.list_docs(self.q, self.like, self.limit, int(self.offset))
            new_rows = page.get("rows", []) if isinstance(page, dict) else []
            table = self.query_one(DataTable)
            if reset:
                table.clear()
                self.rows = []
            for r in new_rows:
                self.rows.append(r)
                table.add_row(
                    r.get("id", ""),
                    (r.get("uri") or "")[:64],
                    (r.get("title") or "")[:48],
                    str(r.get("size") or 0),
                    f"{float(r.get('score') or 0.0):.3f}",
                    str(r.get("chunk_count") or 0),
                    str(r.get("token_count") or 0),
                )
            # prefetch next page when near end? simple approach: if we loaded a full page, bump offset
            if len(new_rows) >= self.limit:
                self.offset += self.limit
        except Exception as e:
            self.query_one(Preview).update(f"load error: {e}")

    async def action_export_rows(self) -> None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path.cwd() / f"qq_export_{ts}.jsonl"
        try:
            # Export by current query if present; otherwise export currently loaded ids
            ids = [r.get("id", "") for r in self.rows] if not self.q else None
            await self.backend.export(ids=ids, q=self.q, to_path=str(out))
            self.query_one(Preview).update(f"exported to {out}")
        except Exception as e:
            self.query_one(Preview).update(f"export failed: {e}")

    async def action_toggle_backend(self) -> None:
        if self.alt_backend is None:
            return
        self.backend, self.alt_backend = self.alt_backend, self.backend
        self.offset = 0
        await self._load_page(reset=True)


def run(
    snapshot: Optional[str] = None,
    *,
    enable_live: bool = True,
    prefer_snapshot: bool = False,
) -> None:
    """Launch the Textual TUI.

    Parameters
    ----------
    snapshot:
        Optional snapshot database path for offline mode.
    enable_live:
        Whether to connect to the live HTTP API backend.
    prefer_snapshot:
        When both backends are available, choose snapshot as primary.
    """
    live: Backend | None = LiveBackend() if enable_live else None
    snap: Backend | None = SnapshotBackend(snapshot) if snapshot else None

    backend: Backend | None
    alt: Backend | None = None
    if prefer_snapshot and snap is not None:
        backend, alt = snap, live
    elif live is not None:
        backend, alt = live, snap
    elif snap is not None:
        backend, alt = snap, None
    else:
        raise RuntimeError(
            "No backend available for qq tui. Provide --snapshot or ensure the qq API is running."
        )
    app = QQTui(backend=backend, alt_backend=alt)
    app.run()
