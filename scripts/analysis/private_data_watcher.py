#!/usr/bin/env python3
"""
Private dataset watcher with audio alert.
Polls dataset/ directory for new files. When private data arrives,
blares a loud repeating alert until manually killed (Ctrl+C).
Also triggers NOAM pipeline notification.

Usage:
    python scripts/private_data_watcher.py
"""
import json
import os
import subprocess
import sys
import time
import datetime
from pathlib import Path


# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BULLETIN_FILE = PROJECT_ROOT / ".sdd" / "agents" / "BULLETIN.jsonl"
NOAM_QUEUE = PROJECT_ROOT / ".sdd" / "agents" / "noam" / "TASK_QUEUE.jsonl"

POLL_INTERVAL_S = 15  # check every 15 seconds
ALERT_INTERVAL_S = 8  # repeat audio every 8 seconds

# Files that exist in the public dataset — ignore these
PUBLIC_FILES = frozenset([
    "Archive.zip",
    "dataset_documents",
    "public_dataset.json",
])

# ── helpers ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def current_files() -> set[str]:
    """Return set of filenames currently in DATASET_DIR."""
    if not DATASET_DIR.exists():
        return set()
    return {p.name for p in DATASET_DIR.iterdir()}


def post_bulletin(message: str) -> None:
    """Append a bulletin message for agents."""
    try:
        BULLETIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "from": "private_data_watcher",
            "type": "PRIVATE_DATA_ARRIVED",
            "message": message,
        }
        with open(BULLETIN_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log(f"Warning: could not post bulletin: {e}")


def notify_noam(new_files: set[str]) -> None:
    """Push a high-priority task to NOAM's queue."""
    try:
        NOAM_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        task = {
            "task_id": f"private_ingest_{int(time.time())}",
            "priority": "CRITICAL",
            "type": "ingest_private_dataset",
            "description": (
                "PRIVATE DATASET ARRIVED. "
                f"New files in dataset/: {sorted(new_files)}. "
                "Run full enrichment + ingest pipeline immediately. "
                "Rebuild BM25 index. Trigger EYAL eval after ingest completes."
            ),
            "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(NOAM_QUEUE, "a") as f:
            f.write(json.dumps(task) + "\n")
        log(f"✓ Pushed ingest task to NOAM queue")
    except Exception as e:
        log(f"Warning: could not notify NOAM: {e}")


def play_alert(new_files: set[str]) -> None:
    """
    Play a loud repeating audio alert. Loops forever until Ctrl+C.
    Uses macOS `say` and optionally system beeps.
    """
    files_str = ", ".join(sorted(new_files)) if new_files else "unknown files"
    message = (
        f"ATTENTION! Private dataset has arrived! "
        f"New files: {files_str}. "
        f"Wake up! The competition data is ready! "
        f"NOAM pipeline has been notified! Wake up now!"
    )
    log("=" * 70)
    log("🚨 PRIVATE DATASET ARRIVED! 🚨")
    log(f"   New files: {sorted(new_files)}")
    log("=" * 70)
    log("Playing audio alert — press Ctrl+C to stop")

    count = 0
    while True:
        count += 1
        log(f"🔔 Alert #{count} — Private dataset ready!")
        try:
            # macOS say command — loud voice, high urgency
            subprocess.run(
                ["say", "-v", "Samantha", "-r", "180", message],
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            # Fallback: terminal bell
            for _ in range(5):
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(0.3)
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            log("Alert stopped by user.")
            break

        # Pause between alerts
        try:
            time.sleep(ALERT_INTERVAL_S)
        except KeyboardInterrupt:
            log("Alert stopped by user.")
            break


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log("Private dataset watcher started")
    log(f"Watching: {DATASET_DIR}")
    log(f"Poll interval: {POLL_INTERVAL_S}s")
    log(f"Ignoring public files: {PUBLIC_FILES}")

    if not DATASET_DIR.exists():
        log(f"Warning: {DATASET_DIR} does not exist yet — will create and watch")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline: what's already there (public data)
    known_files = current_files()
    log(f"Baseline files: {known_files}")

    while True:
        try:
            present = current_files()
            new_files = present - known_files - PUBLIC_FILES

            if new_files:
                log(f"🆕 NEW FILES DETECTED: {new_files}")
                post_bulletin(f"Private dataset arrived: {sorted(new_files)}")
                notify_noam(new_files)
                play_alert(new_files)
                # After user dismisses alert, keep watching for more files
                known_files = present
                log("Continuing watch for additional files...")
            else:
                log(f"No new files yet. Known: {present}. Sleeping {POLL_INTERVAL_S}s...")

            time.sleep(POLL_INTERVAL_S)

        except KeyboardInterrupt:
            log("Watcher stopped by user.")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
