from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"


def _compose_services() -> dict[str, object]:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    services = payload.get("services")
    assert isinstance(services, dict)
    return cast("dict[str, object]", services)


def _compose_env_file_paths(service_name: str) -> list[str]:
    services = _compose_services()
    service = services[service_name]
    assert isinstance(service, dict)
    env_file = service.get("env_file")
    assert isinstance(env_file, list)
    paths: list[str] = []
    for item in env_file:
        assert isinstance(item, dict)
        path = item.get("path")
        assert isinstance(path, str)
        paths.append(path)
        assert item.get("required") is False
    return paths


def test_compose_tools_and_api_use_container_qdrant_host() -> None:
    services = _compose_services()
    for service_name in ("api", "ingest", "eval"):
        service = services[service_name]
        assert isinstance(service, dict)
        environment = service.get("environment")
        assert isinstance(environment, dict)
        assert environment["QDRANT_URL"] == "http://qdrant:6333"


def test_compose_service_env_file_contract_layers_env_local_after_env() -> None:
    for service_name in ("api", "ingest", "eval"):
        assert _compose_env_file_paths(service_name) == [
            ".env",
            ".env.local",
            "${ENV_FILE:-.env.local}",
        ]
