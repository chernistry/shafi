"""Parse data/page_benchmark_*.md files into timeline data."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .timestamp import extract_label, parse_filename_timestamp

logger = logging.getLogger(__name__)

_FBETA = re.compile(r"F_beta\(2\.5\):\s*([\d.]+)")
_ORPHAN = re.compile(r"Orphan[- ]page case rate:\s*([\d.]+)")
_SLOT_RECALL = re.compile(r"Mean slot recall:\s*([\d.]+)")
_OVERPRUNE = re.compile(r"Overprune violations:\s*(\d+)")
_CASES = re.compile(r"Cases:\s*(\d+)")


from datetime import datetime, timezone

def _parse_single_benchmark(path: Path) -> dict | None:
    """Extract metrics from a single benchmark markdown file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    ts = parse_filename_timestamp(path.name)
    if ts is None:
        try:
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None

    entry: dict = {
        "timestamp": ts.isoformat(),
        "label": extract_label(path.name),
        "filename": path.name,
        "filepath": str(path),
    }

    for name, regex in [
        ("f_beta", _FBETA),
        ("orphan_rate", _ORPHAN),
        ("slot_recall", _SLOT_RECALL),
    ]:
        m = regex.search(text)
        entry[name] = float(m.group(1)) if m else None

    m = _OVERPRUNE.search(text)
    entry["overprune_violations"] = int(m.group(1)) if m else None

    m = _CASES.search(text)
    entry["cases"] = int(m.group(1)) if m else None

    return entry


def parse_benchmark_timeline(data_dirs: list[Path]) -> list[dict]:
    """Scan page_benchmark_*.md files and return sorted timeline."""
    results = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for f in data_dir.rglob("page_benchmark_*.md"):
            entry = _parse_single_benchmark(f)
            if entry:
                results.append(entry)
            else:
                logger.warning("Skipping unparseable benchmark: %s", f.name)

    results.sort(key=lambda e: e["timestamp"])
    return results
