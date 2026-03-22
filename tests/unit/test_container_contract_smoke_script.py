from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import yaml
from scripts import container_contract_smoke as mod

if TYPE_CHECKING:
    from pathlib import Path


def test_build_api_contract_command_runs_inside_packaged_api_service() -> None:
    command = mod.build_api_contract_command()

    assert command[:9] == [
        "docker",
        "compose",
        "-f",
        str(mod.COMPOSE_FILE),
        "run",
        "--rm",
        "--no-deps",
        "--entrypoint",
        "python",
    ]
    assert command[9] == "api"
    assert command[10] == "-c"
    assert "container_contract_ok" in command[11]
    assert "load_prompt(\"ingestion/sac_system\")" in command[11]
    assert "QDRANT_URL" in command[11]


def test_build_ingest_commands_use_tools_profile_and_entrypoint() -> None:
    mount_check = mod.build_ingest_mount_check_command()
    ingest_help = mod.build_ingest_help_command()

    assert mount_check[:6] == [
        "docker",
        "compose",
        "-f",
        str(mod.COMPOSE_FILE),
        "--profile",
        "tools",
    ]
    assert ingest_help[:6] == mount_check[:6]
    assert "rag_challenge.ingestion.pipeline" in ingest_help
    assert "/workspace/dataset/dataset_documents" in mount_check[-1]


def test_main_runs_build_qdrant_and_contract_steps(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    commands: list[list[str]] = []

    def _fake_run(command: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        commands.append(command)
        assert cwd == mod.PROJECT_ROOT
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr("sys.argv", ["container_contract_smoke.py"])

    exit_code = mod.main()

    assert exit_code == 0
    assert commands == [
        mod.build_build_command(),
        mod.build_qdrant_up_command(),
        mod.build_api_contract_command(),
        mod.build_ingest_mount_check_command(),
        mod.build_ingest_help_command(),
    ]


def test_compose_ingest_service_mounts_dataset_read_only() -> None:
    compose = yaml.safe_load(mod.COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(compose, dict)
    services = compose.get("services")
    assert isinstance(services, dict)
    ingest = services.get("ingest")
    assert isinstance(ingest, dict)

    command = ingest.get("command")
    volumes = ingest.get("volumes")

    assert command == [
        "python",
        "-m",
        "rag_challenge.ingestion.pipeline",
        "--doc-dir",
        "/workspace/dataset/dataset_documents",
    ]
    assert "./dataset:/workspace/dataset:ro" in volumes
