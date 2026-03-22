from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_COMPATIBLE_PYTHON = "3.12"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_uv_command(
    command: list[str],
    *,
    project_root: Path | None = None,
    python_version: str | None = None,
) -> list[str]:
    if not command:
        raise ValueError("A command is required")

    root = (project_root or _project_root()).resolve()
    compatible_python = (python_version or os.getenv("BM25_COMPAT_PYTHON") or DEFAULT_COMPATIBLE_PYTHON).strip()
    if not compatible_python:
        raise ValueError("A non-empty compatible Python version is required")

    return [
        "uv",
        "run",
        "--isolated",
        "--python",
        compatible_python,
        "--with-editable",
        str(root),
        "--directory",
        str(root),
        *command,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a BM25 / fastembed-sensitive command in an isolated Python 3.12+ environment. "
            "Use this for sparse-retrieval probes that crash under the project Python 3.14 path."
        )
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        default=None,
        help=f"Compatible Python interpreter version to use (default: {DEFAULT_COMPATIBLE_PYTHON})",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute after `--`, for example: -- python scripts/run_candidate_ceiling_cycle.py ...",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("A command is required after `--`")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_uv_command(args.command, python_version=args.python_version)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
