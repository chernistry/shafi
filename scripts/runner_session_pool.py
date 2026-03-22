from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LeaseRecord:
    lease: str
    cmd: list[str]
    cwd: str
    returncode: int | None
    elapsed_s: float
    status: str
    cleanup_performed: bool
    timeout_s: float | None


class RunnerSessionPool:
    def __init__(self, *, session_name: str, pool_size: int = 1) -> None:
        if pool_size != 1:
            raise ValueError("RunnerSessionPool only supports pool_size=1 in the current no-fanout eval loop")
        self._session_name = session_name
        self._pool_size = pool_size
        self._leases: list[LeaseRecord] = []
        self._cleanup_performed_count = 0
        self._closed = False

    def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        lease: str,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("RunnerSessionPool is already closed")
        started = time.perf_counter()
        proc: subprocess.Popen[bytes] | None = None
        cleanup_performed = False
        status = "ok"
        returncode: int | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                start_new_session=True,
            )
            returncode = proc.wait(timeout=timeout_s)
            if returncode != 0:
                status = "error"
                raise subprocess.CalledProcessError(returncode, cmd)
        except subprocess.TimeoutExpired:
            status = "timeout"
            if proc is not None:
                cleanup_performed = self._terminate_process_group(proc) or cleanup_performed
            raise
        except Exception:
            status = "error"
            if proc is not None and proc.poll() is None:
                cleanup_performed = self._terminate_process_group(proc) or cleanup_performed
            raise
        finally:
            if proc is not None and proc.poll() is None:
                cleanup_performed = self._terminate_process_group(proc) or cleanup_performed
            if cleanup_performed:
                self._cleanup_performed_count += 1
            if proc is not None and returncode is None:
                returncode = proc.returncode
            self._leases.append(
                LeaseRecord(
                    lease=lease,
                    cmd=list(cmd),
                    cwd=str(cwd),
                    returncode=returncode,
                    elapsed_s=round(max(0.0, time.perf_counter() - started), 4),
                    status=status,
                    cleanup_performed=cleanup_performed,
                    timeout_s=timeout_s,
                )
            )

    def summary(self) -> dict[str, object]:
        return {
            "session_name": self._session_name,
            "mode": "serialized_pool_size_1",
            "pool_size": self._pool_size,
            "lease_count": len(self._leases),
            "cleanup_performed_count": self._cleanup_performed_count,
            "leases": [asdict(record) for record in self._leases],
            "closed": self._closed,
        }

    def write_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self._closed = True

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen[bytes]) -> bool:
        if proc.poll() is not None:
            return False
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows fallback
                proc.terminate()
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows fallback
                    proc.kill()
                proc.wait(timeout=0.2)
            return True
        except ProcessLookupError:
            return True
