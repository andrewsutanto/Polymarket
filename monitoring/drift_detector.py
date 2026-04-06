"""Live vs backtest performance drift detection.

Tracks rolling live performance and compares to OOS backtest benchmarks.
Fires alerts and auto-degrades when reality diverges from expectations.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HALT = "HALT"


@dataclass
class DriftAlert:
    level: AlertLevel
    metric: str
    live_value: float
    benchmark_value: float
    degradation_pct: float
    message: str


@dataclass
class DriftBenchmarks:
    """OOS backtest benchmarks to compare live performance against."""

    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    max_drawdown: float = 0.0


class DriftDetector:
    """Detect when live performance drifts from backtest expectations."""

    def __init__(
        self,
        benchmarks: DriftBenchmarks | None = None,
        window_size: int = 50,
        warning_threshold: float = 0.25,
        critical_threshold: float = 0.50,
        halt_threshold: float = 2.0,
    ) -> None:
        self._benchmarks = benchmarks or DriftBenchmarks()
        self._window = window_size
        self._warn_thresh = warning_threshold
        self._crit_thresh = critical_threshold
        self._halt_thresh = halt_threshold

        self._trade_pnls: deque[float] = deque(maxlen=window_size)
        self._trade_edges: deque[float] = deque(maxlen=window_size)
        self._equity_curve: list[float] = []
        self._alert_callbacks: list[Callable[[DriftAlert], Any]] = []
        self._position_scale: float = 1.0  # 1.0 = normal, 0.5 = degraded

    @property
    def position_scale(self) -> float:
        """Current position sizing multiplier (1.0 normal, 0.5 degraded)."""
        return self._position_scale

    def set_benchmarks(self, benchmarks: DriftBenchmarks) -> None:
        self._benchmarks = benchmarks

    def on_alert(self, callback: Callable[[DriftAlert], Any]) -> None:
        self._alert_callbacks.append(callback)

    def record_trade(self, pnl: float, edge: float, equity: float) -> list[DriftAlert]:
        """Record a live trade result and check for drift.

        Args:
            pnl: Realized P&L of the trade.
            edge: Edge at entry.
            equity: Current portfolio equity.

        Returns:
            List of alerts triggered by this trade.
        """
        self._trade_pnls.append(pnl)
        self._trade_edges.append(edge)
        self._equity_curve.append(equity)

        if len(self._trade_pnls) < 10:
            return []

        return self._check_all()

    def _check_all(self) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        pnls = np.array(self._trade_pnls)

        # Win rate
        live_wr = np.sum(pnls > 0) / len(pnls)
        alerts.extend(self._check_metric("win_rate", live_wr, self._benchmarks.win_rate))

        # Rolling Sharpe
        if np.std(pnls) > 1e-10:
            live_sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(365)
        else:
            live_sharpe = 0.0
        alerts.extend(self._check_metric("sharpe", live_sharpe, self._benchmarks.sharpe))

        # Average edge
        live_edge = float(np.mean(self._trade_edges))
        alerts.extend(self._check_metric("avg_edge", live_edge, self._benchmarks.avg_edge))

        # Drawdown
        if self._equity_curve:
            ec = np.array(self._equity_curve)
            peak = np.maximum.accumulate(ec)
            dd = np.max((peak - ec) / np.where(peak > 0, peak, 1.0))
            bench_dd = self._benchmarks.max_drawdown

            if bench_dd > 0:
                if dd >= bench_dd * self._halt_thresh:
                    alerts.append(DriftAlert(
                        AlertLevel.HALT, "max_drawdown", float(dd), bench_dd,
                        (dd / bench_dd - 1) * 100,
                        f"Drawdown {dd:.1%} exceeds {self._halt_thresh}x OOS max ({bench_dd:.1%})",
                    ))
                elif dd >= bench_dd * 1.5:
                    alerts.append(DriftAlert(
                        AlertLevel.CRITICAL, "max_drawdown", float(dd), bench_dd,
                        (dd / bench_dd - 1) * 100,
                        f"Drawdown {dd:.1%} exceeds 1.5x OOS max ({bench_dd:.1%})",
                    ))

        # HALT check: negative Sharpe over full window
        if len(pnls) >= self._window and live_sharpe < 0:
            alerts.append(DriftAlert(
                AlertLevel.HALT, "sharpe_negative", live_sharpe, 0.0, 100.0,
                f"Sharpe negative ({live_sharpe:.2f}) over {self._window} trades — halting",
            ))

        # Apply position scaling
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                self._position_scale = 0.5
            elif alert.level == AlertLevel.HALT:
                self._position_scale = 0.0

        # Fire callbacks
        for alert in alerts:
            for cb in self._alert_callbacks:
                try:
                    cb(alert)
                except Exception:
                    logger.exception("Drift alert callback error")

        return alerts

    def _check_metric(
        self, name: str, live: float, benchmark: float
    ) -> list[DriftAlert]:
        if benchmark <= 0:
            return []
        degradation = (benchmark - live) / benchmark
        if degradation >= self._crit_thresh:
            return [DriftAlert(
                AlertLevel.CRITICAL, name, live, benchmark, degradation * 100,
                f"{name} degraded {degradation:.0%}: live={live:.3f} vs OOS={benchmark:.3f}",
            )]
        if degradation >= self._warn_thresh:
            return [DriftAlert(
                AlertLevel.WARNING, name, live, benchmark, degradation * 100,
                f"{name} degraded {degradation:.0%}: live={live:.3f} vs OOS={benchmark:.3f}",
            )]
        return []
