"""Parse .sdd/researches/ ticket directories for the Research Explorer."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .timestamp import parse_iso_date

logger = logging.getLogger(__name__)

_TICKET_RE = re.compile(r"^(\d+)_(.+?)(?:_r\d+)?(?:_\d{4}-\d{2}-\d{2})?$")


def parse_tickets(researches_dir: Path) -> list[dict]:
    """Scan ticket directories and return metadata list."""
    if not researches_dir.is_dir():
        return []

    results = []
    for d in sorted(researches_dir.iterdir()):
        if not d.is_dir():
            continue

        m = _TICKET_RE.match(d.name)
        if not m:
            continue

        ticket_id = m.group(1)
        title = m.group(2).replace("_", " ")
        date = parse_iso_date(d.name)

        # Check for closeout
        closeout = d / "closeout.md"
        closeout_text = ""
        status = "open"
        if closeout.exists():
            try:
                closeout_text = closeout.read_text(encoding="utf-8")[:2000]
                status = "closed"
            except OSError:
                pass

        results.append({
            "ticket_id": ticket_id,
            "title": title,
            "date": date.isoformat() if date else None,
            "dir_name": d.name,
            "status": status,
            "closeout_summary": closeout_text[:500] if closeout_text else "",
            "file_count": sum(1 for _ in d.iterdir()) if d.is_dir() else 0,
        })

    results.sort(key=lambda e: e.get("date") or "", reverse=True)
    return results
