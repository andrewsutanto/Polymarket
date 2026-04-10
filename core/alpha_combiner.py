"""Alpha Combination Engine — institutional signal combining.

Implements the Fundamental Law of Active Management:
    IR = IC × √N

Where:
    IR = Information Ratio (risk-adjusted edge)
    IC = average Information Coefficient per signal
    N  = number of truly independent signals

Tracks per-strategy IC, computes signal correlation matrix,
derives effective-N via eigenvalue decomposition, and produces
optimal weights via mean-variance optimization.

Replaces naive static weights in ensemble.py with data-driven
weights that adapt as the system learns from trade outcomes.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Rolling performance metrics for a single strategy."""

    name: str
    ic: float = 0.0           # Information Coefficient
    win_rate: float = 0.0
    avg_edge: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    optimal_weight: float = 0.0
    predictions: deque = field(default_factory=lambda: deque(maxlen=200))
    outcomes: deque = field(default_factory=lambda: deque(maxlen=200))
    returns: deque = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class CombinedSignal:
    """Output of the alpha combination engine."""

    direction: str
    combined_edge: float
    combined_strength: float
    effective_n: float
    information_ratio: float
    weights_used: dict[str, float]
    contributing_signals: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


class AlphaCombiner:
    """Institutional-grade signal combination engine.

    Tracks IC per strategy, computes correlation matrix,
    and derives optimal weights using the Fundamental Law.
    """

    def __init__(
        self,
        min_trades_for_ic: int = 20,
        ic_window: int = 100,
        regularization: float = 0.1,
        min_ic_threshold: float = -0.05,
    ) -> None:
        self._min_trades = min_trades_for_ic
        self._ic_window = ic_window
        self._regularization = regularization
        self._min_ic = min_ic_threshold

        self._strategies: dict[str, StrategyMetrics] = {}
        self._correlation_matrix: np.ndarray | None = None
        self._effective_n: float = 0.0
        self._last_ir: float = 0.0

    @property
    def effective_n(self) -> float:
        return self._effective_n

    @property
    def information_ratio(self) -> float:
        return self._last_ir

    def register_strategy(self, name: str) -> None:
        """Register a strategy for tracking."""
        if name not in self._strategies:
            self._strategies[name] = StrategyMetrics(name=name)

    def record_prediction(
        self, strategy_name: str, predicted_edge: float, direction: str
    ) -> None:
        """Record a strategy's prediction (before outcome is known)."""
        if strategy_name not in self._strategies:
            self.register_strategy(strategy_name)

        signed = predicted_edge if direction == "BUY" else -predicted_edge
        self._strategies[strategy_name].predictions.append(signed)

    def record_outcome(
        self, strategy_name: str, actual_return: float
    ) -> None:
        """Record the actual return after a trade resolves.

        This updates the strategy's IC and performance metrics.
        """
        if strategy_name not in self._strategies:
            return

        metrics = self._strategies[strategy_name]
        metrics.outcomes.append(actual_return)
        metrics.returns.append(actual_return)
        metrics.n_trades += 1
        if actual_return > 0:
            metrics.n_wins += 1
        metrics.win_rate = metrics.n_wins / max(metrics.n_trades, 1)

        # Update IC if enough data
        if len(metrics.predictions) >= self._min_trades and len(metrics.outcomes) >= self._min_trades:
            metrics.ic = self._compute_ic(metrics)
            metrics.avg_edge = float(np.mean(list(metrics.returns)))

        # Recompute correlation and weights after each outcome
        if metrics.n_trades % 5 == 0:
            self._update_correlation_matrix()
            self._compute_optimal_weights()

    def get_optimal_weights(self) -> dict[str, float]:
        """Return current optimal weights for all strategies."""
        weights = {}
        for name, m in self._strategies.items():
            weights[name] = m.optimal_weight
        return weights

    def get_strategy_metrics(self) -> dict[str, dict[str, Any]]:
        """Return current metrics for all strategies."""
        result = {}
        for name, m in self._strategies.items():
            result[name] = {
                "ic": m.ic,
                "win_rate": m.win_rate,
                "avg_edge": m.avg_edge,
                "n_trades": m.n_trades,
                "optimal_weight": m.optimal_weight,
            }
        return result

    def combine_signals(
        self,
        signals: dict[str, dict[str, Any]],
    ) -> CombinedSignal | None:
        """Combine multiple strategy signals using optimal weights.

        Args:
            signals: {strategy_name: {"direction": str, "edge": float, "strength": float, ...}}

        Returns:
            CombinedSignal or None if no valid signals.
        """
        if not signals:
            return None

        weights = self.get_optimal_weights()

        # If we don't have enough data, use equal weights
        if not weights or all(w == 0 for w in weights.values()):
            n = len(signals)
            weights = {name: 1.0 / n for name in signals}

        # Compute weighted directional edge
        buy_edge = 0.0
        sell_edge = 0.0
        buy_weight = 0.0
        sell_weight = 0.0

        for name, sig in signals.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue

            edge = sig.get("edge", 0.0)
            strength = sig.get("strength", 0.5)

            if sig.get("direction") == "BUY":
                buy_edge += edge * w * strength
                buy_weight += w
            else:
                sell_edge += edge * w * strength
                sell_weight += w

        if buy_edge == 0 and sell_edge == 0:
            return None

        if buy_edge >= sell_edge:
            direction = "BUY"
            combined_edge = buy_edge / max(buy_weight, 1e-8)
            combined_strength = buy_weight / (buy_weight + sell_weight) if (buy_weight + sell_weight) > 0 else 0.5
        else:
            direction = "SELL"
            combined_edge = sell_edge / max(sell_weight, 1e-8)
            combined_strength = sell_weight / (buy_weight + sell_weight) if (buy_weight + sell_weight) > 0 else 0.5

        # Compute IR
        avg_ic = np.mean([m.ic for m in self._strategies.values() if m.n_trades >= self._min_trades]) if self._strategies else 0.0
        ir = avg_ic * np.sqrt(max(self._effective_n, 1.0))
        self._last_ir = float(ir)

        return CombinedSignal(
            direction=direction,
            combined_edge=round(combined_edge, 4),
            combined_strength=round(min(combined_strength, 1.0), 4),
            effective_n=round(self._effective_n, 2),
            information_ratio=round(ir, 4),
            weights_used={k: round(v, 4) for k, v in weights.items() if k in signals},
            contributing_signals=signals,
            metadata={
                "n_strategies_total": len(self._strategies),
                "n_strategies_active": len(signals),
                "avg_ic": round(avg_ic, 4),
                "buy_edge": round(buy_edge, 4),
                "sell_edge": round(sell_edge, 4),
            },
        )

    def _compute_ic(self, metrics: StrategyMetrics) -> float:
        """Compute Information Coefficient (rank correlation of predictions vs outcomes)."""
        preds = list(metrics.predictions)
        outs = list(metrics.outcomes)
        n = min(len(preds), len(outs))
        if n < self._min_trades:
            return 0.0

        preds = np.array(preds[-n:])
        outs = np.array(outs[-n:])

        # Rank correlation (Spearman)
        if np.std(preds) < 1e-10 or np.std(outs) < 1e-10:
            return 0.0

        pred_ranks = np.argsort(np.argsort(preds)).astype(float)
        out_ranks = np.argsort(np.argsort(outs)).astype(float)

        correlation = np.corrcoef(pred_ranks, out_ranks)[0, 1]
        return float(np.nan_to_num(correlation, 0.0))

    def _update_correlation_matrix(self) -> None:
        """Compute correlation matrix between strategy returns."""
        active = [
            (name, m) for name, m in self._strategies.items()
            if len(m.returns) >= self._min_trades
        ]

        if len(active) < 2:
            self._effective_n = float(len(active))
            return

        # Build return matrix
        min_len = min(len(m.returns) for _, m in active)
        returns_matrix = np.array([
            list(m.returns)[-min_len:] for _, m in active
        ])

        if returns_matrix.shape[1] < self._min_trades:
            return

        # Correlation matrix
        try:
            corr = np.corrcoef(returns_matrix)
            corr = np.nan_to_num(corr, nan=0.0)
            self._correlation_matrix = corr

            # Effective N via eigenvalue decomposition
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.maximum(eigenvalues, 0)
            if eigenvalues.sum() > 0:
                # Effective N = (sum of eigenvalues)^2 / sum of eigenvalues^2
                self._effective_n = float(
                    eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()
                )
            else:
                self._effective_n = 1.0

        except (np.linalg.LinAlgError, ValueError):
            self._effective_n = float(len(active))

    def _compute_optimal_weights(self) -> None:
        """Compute optimal weights proportional to IC / noise.

        Weight ∝ IC_i / σ_i, then normalize.
        With regularization toward equal weights to prevent
        concentration in low-data strategies.
        """
        active = {
            name: m for name, m in self._strategies.items()
            if m.n_trades >= self._min_trades
        }

        if not active:
            # Equal weights for all registered strategies
            n = len(self._strategies)
            for m in self._strategies.values():
                m.optimal_weight = 1.0 / max(n, 1)
            return

        # Compute raw weights: IC / volatility
        raw_weights = {}
        for name, m in active.items():
            returns_arr = np.array(list(m.returns))
            vol = np.std(returns_arr) if len(returns_arr) > 1 else 1.0
            vol = max(vol, 0.001)

            if m.ic < self._min_ic:
                raw_weights[name] = 0.0
            else:
                raw_weights[name] = max(m.ic / vol, 0.0)

        # Regularize toward equal weights
        n = len(active)
        equal_weight = 1.0 / n
        reg = self._regularization

        for name in raw_weights:
            raw_weights[name] = (1 - reg) * raw_weights[name] + reg * equal_weight

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            for name in raw_weights:
                raw_weights[name] /= total
        else:
            for name in raw_weights:
                raw_weights[name] = equal_weight

        # Apply to strategies
        for name, m in self._strategies.items():
            m.optimal_weight = raw_weights.get(name, 0.0)

    def get_fundamental_law_stats(self) -> dict[str, Any]:
        """Return current Fundamental Law of Active Management stats."""
        active_ics = [
            m.ic for m in self._strategies.values()
            if m.n_trades >= self._min_trades
        ]

        avg_ic = float(np.mean(active_ics)) if active_ics else 0.0
        ir = avg_ic * np.sqrt(max(self._effective_n, 1.0))

        return {
            "avg_ic": round(avg_ic, 4),
            "effective_n": round(self._effective_n, 2),
            "information_ratio": round(ir, 4),
            "n_strategies_tracked": len(self._strategies),
            "n_strategies_with_ic": len(active_ics),
            "per_strategy_ic": {
                name: round(m.ic, 4)
                for name, m in self._strategies.items()
                if m.n_trades >= self._min_trades
            },
            "optimal_weights": {
                name: round(m.optimal_weight, 4)
                for name, m in self._strategies.items()
            },
            "correlation_matrix": self._correlation_matrix.tolist() if self._correlation_matrix is not None else None,
        }
