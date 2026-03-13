from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_bm25_compatible_command as mod


def test_build_uv_command_uses_isolated_compatible_runtime(tmp_path: Path) -> None:
    command = mod.build_uv_command(
        ["python", "scripts/run_candidate_ceiling_cycle.py", "--dry-run"],
        project_root=tmp_path,
        python_version="3.13",
    )

    assert command == [
        "uv",
        "run",
        "--isolated",
        "--python",
        "3.13",
        "--with-editable",
        str(tmp_path.resolve()),
        "--directory",
        str(tmp_path.resolve()),
        "python",
        "scripts/run_candidate_ceiling_cycle.py",
        "--dry-run",
    ]


def test_build_uv_command_rejects_missing_command() -> None:
    with pytest.raises(ValueError, match="command is required"):
        mod.build_uv_command([])


def test_main_executes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        calls.append(command)
        assert check is False
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    exit_code = mod.main(["--python", "3.12", "--", "python", "-c", "print('ok')"])

    assert exit_code == 7
    assert calls == [
        [
            "uv",
            "run",
            "--isolated",
            "--python",
            "3.12",
            "--with-editable",
            str(Path(mod.__file__).resolve().parents[1]),
            "--directory",
            str(Path(mod.__file__).resolve().parents[1]),
            "python",
            "-c",
            "print('ok')",
        ]
    ]
