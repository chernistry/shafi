from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str = "default"
    failure_threshold: int = 3
    reset_timeout_s: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker '%s' recovered -> CLOSED", self.name)
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker '%s' OPEN after %d failures",
                    self.name,
                    self.failure_count,
                )
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.reset_timeout_s:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker '%s' -> HALF_OPEN", self.name)
                return True
            return False
        return True
