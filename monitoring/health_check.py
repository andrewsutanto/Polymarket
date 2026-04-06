"""Data quality and system health checks.

Monitors for stale data, forecast anomalies, spread anomalies,
execution quality, and regime instability.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from monitoring.drift_detector import AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class HealthAlert:
    level: AlertLevel
    check: str
    message: str
    timestamp: datetime


class HealthCheck:
    """Monitor data quality and system health."""

    def __init__(
        self,
        stale_threshold_sec: float = 300,
        forecast_anomaly_sigma: float = 3.0,
        spread_anomaly_sigma: float = 5.0,
        slippage_alert_ratio: float = 2.0,
        regime_change_max_per_hour: int = 3,
        expected_slippage: float = 0.005,
    ) -> None:
        self._stale_sec = stale_threshold_sec
        self._fc_sigma = forecast_anomaly_sigma
        self._spread_sigma = spread_anomaly_sigma
        self._slip_ratio = slippage_alert_ratio
        self._regime_max = regime_change_max_per_hour
        self._expected_slippage = expected_slippage

        # State
        self._last_noaa_update: datetime | None = None
        self._last_pm_update: datetime | None = None
        self._forecast_changes: deque[float] = deque(maxlen=100)
        self._spread_history: deque[float] = deque(maxlen=100)
        self._slippage_history: deque[float] = deque(maxlen=20)
        self._regime_changes: deque[datetime] = deque(maxlen=20)
        self._alerts: list[HealthAlert] = []

    @property
    def recent_alerts(self) -> list[HealthAlert]:
        return list(self._alerts[-20:])

    def record_noaa_update(self, timestamp: datetime) -> None:
        self._last_noaa_update = timestamp

    def record_pm_update(self, timestamp: datetime) -> None:
        self._last_pm_update = timestamp

    def record_forecast_change(self, shift_f: float) -> None:
        self._forecast_changes.append(abs(shift_f))

    def record_spread(self, spread: float) -> None:
        self._spread_history.append(spread)

    def record_slippage(self, realized: float) -> None:
        self._slippage_history.append(abs(realized))

    def record_regime_change(self, timestamp: datetime) -> None:
        self._regime_changes.append(timestamp)

    def run_checks(self, now: datetime | None = None) -> list[HealthAlert]:
        """Run all health checks and return any alerts.

        Args:
            now: Current time (for testing).

        Returns:
            List of HealthAlerts fired this check.
        """
        now = now or datetime.now(timezone.utc)
        alerts: list[HealthAlert] = []

        # Stale data
        if self._last_noaa_update:
            elapsed = (now - self._last_noaa_update).total_seconds()
            if elapsed > self._stale_sec:
                alerts.append(HealthAlert(
                    AlertLevel.WARNING, "stale_noaa",
                    f"NOAA data stale: {elapsed:.0f}s since last update",
                    now,
                ))

        if self._last_pm_update:
            elapsed = (now - self._last_pm_update).total_seconds()
            if elapsed > self._stale_sec:
                alerts.append(HealthAlert(
                    AlertLevel.WARNING, "stale_polymarket",
                    f"Polymarket data stale: {elapsed:.0f}s since last update",
                    now,
                ))

        # Forecast anomaly
        if len(self._forecast_changes) >= 10:
            arr = np.array(self._forecast_changes)
            mean, std = np.mean(arr), np.std(arr)
            if std > 0 and len(arr) > 0:
                latest = arr[-1]
                if latest > mean + self._fc_sigma * std:
                    alerts.append(HealthAlert(
                        AlertLevel.CRITICAL, "forecast_anomaly",
                        f"Forecast change {latest:.1f}°F exceeds {self._fc_sigma}σ "
                        f"(mean={mean:.1f}, σ={std:.1f})",
                        now,
                    ))

        # Spread anomaly
        if len(self._spread_history) >= 20:
            arr = np.array(self._spread_history)
            mean, std = np.mean(arr), np.std(arr)
            if std > 0:
                latest = arr[-1]
                if abs(latest - mean) > self._spread_sigma * std:
                    alerts.append(HealthAlert(
                        AlertLevel.WARNING, "spread_anomaly",
                        f"Spread {latest:.4f} exceeds {self._spread_sigma}σ from mean {mean:.4f}",
                        now,
                    ))

        # Execution quality
        if len(self._slippage_history) >= 10:
            avg_slip = float(np.mean(self._slippage_history))
            if avg_slip > self._expected_slippage * self._slip_ratio:
                alerts.append(HealthAlert(
                    AlertLevel.WARNING, "slippage_high",
                    f"Avg slippage {avg_slip:.4f} exceeds {self._slip_ratio}x expected ({self._expected_slippage:.4f})",
                    now,
                ))

        # Regime instability
        cutoff = now - timedelta(hours=1)
        recent_changes = [t for t in self._regime_changes if t > cutoff]
        if len(recent_changes) > self._regime_max:
            alerts.append(HealthAlert(
                AlertLevel.WARNING, "regime_instability",
                f"{len(recent_changes)} regime changes in last hour (max={self._regime_max})",
                now,
            ))

        self._alerts.extend(alerts)
        return alerts
