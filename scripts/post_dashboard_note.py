#!/usr/bin/env python3
"""Post a status note to the dashboard. Usage: python3 scripts/post_dashboard_note.py "Note text" """
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

NOTES_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "static" / "status_notes.json"
MAX_NOTES = 30  # Keep last 30 notes


def post_note(text: str, author: str = "noam") -> None:
    notes: list[dict] = []
    if NOTES_PATH.exists():
        try:
            notes = json.loads(NOTES_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            notes = []

    notes.insert(0, {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "note": text,
        "author": author,
    })

    # Trim to MAX_NOTES
    notes = notes[:MAX_NOTES]
    NOTES_PATH.write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n")
    print(f"Posted ({len(text)} chars, {len(notes)} total notes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/post_dashboard_note.py 'Note text' [author]")
        sys.exit(1)

    text = sys.argv[1]
    author = sys.argv[2] if len(sys.argv) > 2 else "noam"
    post_note(text, author)
