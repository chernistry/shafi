"""Shared timestamp extraction from filenames."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_TS_8_4 = re.compile(r"(\d{8})_(\d{4})")  # ddmmyyyy_hhmm
_TS_8_6 = re.compile(r"(\d{8})_(\d{6})")  # ddmmyyyy_hhmmss
_TS_6_6 = re.compile(r"(\d{6})_(\d{6})")  # ddmmyy_hhmmss
_TS_6_0 = re.compile(r"(\d{6})_")         # ddmmyy_
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_filename_timestamp(name: str) -> datetime | None:
    """Extract a datetime from an eval/judge/benchmark filename."""
    m6_6 = _TS_6_6.search(name)
    if m6_6:
        try:
            return datetime.strptime(m6_6.group(1) + m6_6.group(2), "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
            
    m6 = _TS_8_6.search(name)
    if m6:
        try:
            return datetime.strptime(m6.group(1) + m6.group(2), "%d%m%Y%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
            
    m4 = _TS_8_4.search(name)
    if m4:
        try:
            return datetime.strptime(m4.group(1) + m4.group(2), "%d%m%Y%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
            
    m6_0 = _TS_6_0.search(name)
    if m6_0:
        try:
            return datetime.strptime(m6_0.group(1), "%d%m%y").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
            
    return None


def parse_iso_date(name: str) -> datetime | None:
    """Extract ISO date (yyyy-mm-dd) from a directory/file name."""
    m = _ISO_DATE.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def extract_label(name: str) -> str:
    """Extract a human-readable label from filename after the timestamp."""
    # Remove extension
    base = re.sub(r"\.\w+$", "", name)
    # Remove known prefixes
    base = re.sub(r"^(eval|judge|page_benchmark|submission_projection)_", "", base)
    # Remove timestamp portion
    base = _TS_8_6.sub("", base)
    base = _TS_8_4.sub("", base)
    base = _TS_6_6.sub("", base)
    base = _TS_6_0.sub("", base)
    # Clean up
    label = base.strip("_ ").replace("_", " ")
    return label or ""
