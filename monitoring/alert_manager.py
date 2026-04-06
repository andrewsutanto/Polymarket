"""Severity-based alert manager with deduplication and daily digest.

Integrates with existing Telegram service for delivery.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Awaitable

from monitoring.drift_detector import AlertLevel, DriftAlert
from monitoring.health_check import HealthAlert

logger = logging.getLogger(__name__)


class AlertManager:
    """Manage, deduplicate, and dispatch alerts."""

    def __init__(
        self,
        cooldown_sec: float = 300,
        send_fn: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self._cooldown = timedelta(seconds=cooldown_sec)
        self._send_fn = send_fn
        self._last_sent: dict[str, datetime] = {}
        self._daily_log: list[dict[str, Any]] = []
        self._counts: dict[str, int] = defaultdict(int)

    def set_sender(self, fn: Callable[[str], Awaitable[bool]]) -> None:
        """Set the async message sender (e.g., telegram.send_message)."""
        self._send_fn = fn

    async def handle_drift_alert(self, alert: DriftAlert) -> bool:
        """Process a drift alert."""
        key = f"drift_{alert.metric}_{alert.level.value}"
        self._daily_log.append({
            "type": "drift",
            "level": alert.level.value,
            "metric": alert.metric,
            "message": alert.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._counts[alert.level.value] += 1

        if not self._should_send(key):
            return False

        emoji = _level_emoji(alert.level)
        text = (
            f"{emoji} <b>DRIFT {alert.level.value}</b>\n\n"
            f"{alert.message}\n"
            f"Live: {alert.live_value:.4f} | Benchmark: {alert.benchmark_value:.4f}\n"
            f"Degradation: {alert.degradation_pct:.1f}%"
        )
        return await self._send(text)

    async def handle_health_alert(self, alert: HealthAlert) -> bool:
        """Process a health check alert."""
        key = f"health_{alert.check}_{alert.level.value}"
        self._daily_log.append({
            "type": "health",
            "level": alert.level.value,
            "check": alert.check,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
        })
        self._counts[alert.level.value] += 1

        if not self._should_send(key):
            return False

        emoji = _level_emoji(alert.level)
        text = f"{emoji} <b>HEALTH {alert.level.value}</b>\n\n{alert.message}"
        return await self._send(text)

    async def send_daily_digest(
        self,
        regime_breakdown: dict[str, float] | None = None,
        strategy_pnl: dict[str, float] | None = None,
        city_pnl: dict[str, float] | None = None,
        drift_metrics: dict[str, float] | None = None,
    ) -> bool:
        """Send daily summary at midnight UTC."""
        lines = ["\U0001f4cb <b>DAILY DIGEST</b>\n"]

        # Alert counts
        total = sum(self._counts.values())
        lines.append(f"Alerts today: {total}")
        for level in ("HALT", "CRITICAL", "WARNING", "INFO"):
            c = self._counts.get(level, 0)
            if c:
                lines.append(f"  {_level_emoji(AlertLevel(level))} {level}: {c}")

        # Regime
        if regime_breakdown:
            lines.append("\n<b>Regime Breakdown:</b>")
            for r, pct in sorted(regime_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  {r}: {pct:.0%}")

        # Strategy P&L
        if strategy_pnl:
            lines.append("\n<b>Strategy P&L:</b>")
            for s, pnl in strategy_pnl.items():
                emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
                lines.append(f"  {emoji} {s}: ${pnl:+.4f}")

        # City P&L
        if city_pnl:
            lines.append("\n<b>City P&L:</b>")
            for c, pnl in city_pnl.items():
                lines.append(f"  {c}: ${pnl:+.4f}")

        # Drift
        if drift_metrics:
            lines.append("\n<b>Drift Metrics:</b>")
            for m, v in drift_metrics.items():
                lines.append(f"  {m}: {v:.4f}")

        # Reset daily
        self._daily_log.clear()
        self._counts.clear()

        return await self._send("\n".join(lines))

    def _should_send(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self._last_sent.get(key)
        if last and (now - last) < self._cooldown:
            return False
        self._last_sent[key] = now
        return True

    async def _send(self, text: str) -> bool:
        if self._send_fn:
            try:
                return await self._send_fn(text)
            except Exception as exc:
                logger.error("Alert send failed: %s", exc)
        else:
            logger.info("[ALERT] %s", text[:200])
        return False


def _level_emoji(level: AlertLevel) -> str:
    return {
        AlertLevel.INFO: "\u2139\ufe0f",
        AlertLevel.WARNING: "\u26a0\ufe0f",
        AlertLevel.CRITICAL: "\U0001f6a8",
        AlertLevel.HALT: "\U0001f6d1",
    }.get(level, "\u2753")
