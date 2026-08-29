"""Vendor quota constants and pacing gate."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from src.data.providers.base import ProviderError


@dataclass(frozen=True, slots=True)
class VendorQuota:
    requests_per_hour: int
    requests_per_day: int
    min_interval_s: float


TIINGO_QUOTA: Final[VendorQuota] = VendorQuota(
    requests_per_hour=50,
    requests_per_day=1000,
    min_interval_s=float(math.ceil(3600 / 50)),
)


class PacingGate:
    """Enforce min interval and daily budget before each HTTP GET."""

    def __init__(
        self,
        quota: VendorQuota,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._quota = quota
        self._clock: Callable[[], float] = clock if clock is not None else time.time
        self._sleeper: Callable[[float], None] = sleeper if sleeper is not None else time.sleep
        self._last_acquire: float | None = None
        self._acquires_today: int = 0
        self._today: object | None = None

    def acquire(self) -> None:
        now = self._clock()
        # Derive UTC calendar day from timestamp.
        try:
            today: object = datetime.fromtimestamp(now, tz=UTC).date()
        except (OSError, OverflowError, ValueError):
            # Fallback for fake clocks with small epoch values.
            today = int(now // 86400)
        if today != self._today:
            self._today = today
            self._acquires_today = 0
        if self._acquires_today >= self._quota.requests_per_day:
            raise ProviderError("rate limit exceeded: daily quota exhausted")
        if self._last_acquire is not None:
            elapsed = now - self._last_acquire
            needed = self._quota.min_interval_s - elapsed
            if needed > 0:
                self._sleeper(needed)
                now = self._clock()
                # Re-evaluate day after sleep (cross midnight).
                try:
                    today2: object = datetime.fromtimestamp(now, tz=UTC).date()
                except (OSError, OverflowError, ValueError):
                    today2 = int(now // 86400)
                if today2 != self._today:
                    self._today = today2
                    self._acquires_today = 0
                    if self._acquires_today >= self._quota.requests_per_day:
                        raise ProviderError("rate limit exceeded: daily quota exhausted")
        self._acquires_today += 1
        self._last_acquire = self._clock()

    @property
    def remaining_daily(self) -> int:
        return max(0, self._quota.requests_per_day - self._acquires_today)
