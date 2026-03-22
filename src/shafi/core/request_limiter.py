from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

SleepFunc = Callable[[float], Awaitable[None]]
ClockFunc = Callable[[], float]


class AsyncRequestLimiter:
    """Throttle async requests with shared concurrency and start-time spacing.

    Args:
        concurrency_limit: Maximum number of in-flight requests.
        min_interval_s: Minimum spacing between request start times.
        sleep_func: Awaitable sleep implementation.
        clock: Monotonic clock returning seconds.

    Returns:
        None.
    """

    def __init__(
        self,
        concurrency_limit: int,
        min_interval_s: float,
        *,
        sleep_func: SleepFunc = asyncio.sleep,
        clock: ClockFunc = time.monotonic,
    ) -> None:
        self._concurrency_limit = max(1, int(concurrency_limit))
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._sleep = sleep_func
        self._clock = clock
        self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        self._schedule_lock = asyncio.Lock()
        self._next_allowed_start_s = 0.0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a request slot and enforce configured start spacing.

        Args:
            None.

        Returns:
            An async context manager guarding one throttled request.
        """

        await self._semaphore.acquire()
        try:
            await self._wait_for_turn()
            yield
        finally:
            self._semaphore.release()

    async def _wait_for_turn(self) -> None:
        """Delay until the next request start is permitted.

        Args:
            None.

        Returns:
            None.
        """

        if self._min_interval_s <= 0.0:
            return

        async with self._schedule_lock:
            now_s = self._clock()
            due_s = max(now_s, self._next_allowed_start_s)
            self._next_allowed_start_s = due_s + self._min_interval_s

        wait_s = max(0.0, due_s - self._clock())
        if wait_s > 0.0:
            await self._sleep(wait_s)


_LIMITER_REGISTRY: dict[tuple[str, int, float, int, int], AsyncRequestLimiter] = {}


def get_shared_request_limiter(
    name: str,
    *,
    concurrency_limit: int,
    min_interval_s: float,
    sleep_func: SleepFunc = asyncio.sleep,
    clock: ClockFunc = time.monotonic,
) -> AsyncRequestLimiter:
    """Return a process-shared async request limiter.

    Args:
        name: Logical limiter name.
        concurrency_limit: Maximum number of in-flight requests.
        min_interval_s: Minimum spacing between request start times.
        sleep_func: Awaitable sleep implementation.
        clock: Monotonic clock returning seconds.

    Returns:
        Shared limiter instance for the provided configuration.
    """

    normalized_limit = max(1, int(concurrency_limit))
    normalized_interval = max(0.0, float(min_interval_s))
    key = (name, normalized_limit, normalized_interval, id(sleep_func), id(clock))
    limiter = _LIMITER_REGISTRY.get(key)
    if limiter is None:
        limiter = AsyncRequestLimiter(
            normalized_limit,
            normalized_interval,
            sleep_func=sleep_func,
            clock=clock,
        )
        _LIMITER_REGISTRY[key] = limiter
    return limiter


def reset_request_limiter_registry_for_tests() -> None:
    """Clear the shared limiter registry for deterministic tests.

    Args:
        None.

    Returns:
        None.
    """

    _LIMITER_REGISTRY.clear()
