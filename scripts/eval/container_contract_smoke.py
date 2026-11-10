#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
COMPOSE_FILE: Final[Path] = PROJECT_ROOT / "docker-compose.yml"
API_SERVICE: Final[str] = "api"
INGEST_SERVICE: Final[str] = "ingest"
QDRANT_SERVICE: Final[str] = "qdrant"
_CONTAINER_CONTRACT_SCRIPT: Final[str] = """
import os
from pathlib import Path
from urllib.request import urlopen

from shafi.api.app import create_app
from shafi.prompts.loader import load_prompt

app = create_app()
assert app.title == "Legal RAG Challenge 2026"
assert Path.home() == Path("/home/appuser")
assert load_prompt("app/warmup_system")
assert load_prompt("ingestion/sac_system")
assert load_prompt("llm/generator_system_strict")

for env_name in ("XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME"):
    path = Path(os.environ[env_name])
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".container-contract-write"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

health_url = os.environ["QDRANT_URL"].rstrip("/") + "/healthz"
with urlopen(health_url, timeout=5) as response:
    assert response.status == 200

print("container_contract_ok")
""".strip()


def build_compose_command(*args: str, profile_tools: bool = False) -> list[str]:
    """Build a docker compose command rooted at the repo compose file.

    Args:
        *args: Compose CLI arguments to append.
        profile_tools: Whether to include the `tools` profile.

    Returns:
        The full docker compose command.
    """
    command = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    if profile_tools:
        command.extend(["--profile", "tools"])
    command.extend(args)
    return command


def build_build_command() -> list[str]:
    """Build the local API image used by compose services.

    Returns:
        Compose build command for the API image.
    """
    return build_compose_command("build", API_SERVICE)


def build_qdrant_up_command() -> list[str]:
    """Start Qdrant for the container contract smoke.

    Returns:
        Compose command that starts the local Qdrant service.
    """
    return build_compose_command("up", "-d", QDRANT_SERVICE)


def build_api_contract_command() -> list[str]:
    """Run the packaged-image contract checks inside the API container.

    Returns:
        Compose run command that executes the packaged API/container checks.
    """
    return build_compose_command(
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "python",
        API_SERVICE,
        "-c",
        _CONTAINER_CONTRACT_SCRIPT,
    )


def build_ingest_mount_check_command() -> list[str]:
    """Validate the ingest container mount contract.

    Returns:
        Compose run command that checks the dataset mount inside the ingest
        service container.
    """
    return build_compose_command(
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "python",
        INGEST_SERVICE,
        "-c",
        (
            "from pathlib import Path; "
            "dataset = Path('/workspace/dataset/dataset_documents'); "
            "assert dataset.is_dir(), dataset; "
            "print(dataset)"
        ),
        profile_tools=True,
    )


def build_ingest_help_command() -> list[str]:
    """Validate the ingest entrypoint from the built image.

    Returns:
        Compose run command that executes the ingestion CLI help.
    """
    return build_compose_command(
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "python",
        INGEST_SERVICE,
        "-m",
        "shafi.ingestion.pipeline",
        "--help",
        profile_tools=True,
    )


def run_command(command: list[str], *, dry_run: bool) -> int:
    """Run one smoke-step command.

    Args:
        command: Command to execute.
        dry_run: Whether to print without executing.

    Returns:
        Process exit code. `0` when dry-running.
    """
    print("$", " ".join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    """Parse script arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description="Run the packaged-image Docker/container contract smoke.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-build", action="store_true", help="Assume the compose image already exists.")
    parser.add_argument("--skip-qdrant-up", action="store_true", help="Assume Qdrant is already running.")
    return parser.parse_args()


def main() -> int:
    """Run the packaged-image container smoke sequence.

    Returns:
        Process exit status.
    """
    args = _parse_args()
    commands: list[list[str]] = []
    if not bool(args.skip_build):
        commands.append(build_build_command())
    if not bool(args.skip_qdrant_up):
        commands.append(build_qdrant_up_command())
    commands.extend(
        [
            build_api_contract_command(),
            build_ingest_mount_check_command(),
            build_ingest_help_command(),
        ]
    )

    for command in commands:
        exit_code = run_command(command, dry_run=bool(args.dry_run))
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
