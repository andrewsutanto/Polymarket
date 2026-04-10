"""Bayesian strategy weight updater — learn from resolved trades.

After each trade resolves, updates strategy weight priors using
conjugate Beta distributions for win rates and tracks per-strategy
Information Coefficient (IC) over a rolling window.

The posterior weights feed into the AlphaCombiner so the ensemble
adapts to which strategies are actually generating alpha in the
current market regime.

Key concepts:
- Beta(alpha, beta) conjugate prior for Bernoulli win/loss outcomes
- Rolling IC computation via rank correlation
- Posterior-weighted strategy allocation via expected Sharpe
- Regime detection via IC divergence from prior
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyPosterior:
    """Bayesian posterior state for a single strategy."""

    name: str

    # Beta distribution parameters for win rate
    alpha: float = 1.0  # prior successes + 1 (start uniform Beta(1,1))
    beta: float = 1.0   # prior failures + 1

    # Rolling performance tracking
    predictions: deque = field(default_factory=lambda: deque(maxlen=200))
    outcomes: deque = field(default_factory=lambda: deque(maxlen=200))
    returns: deque = field(default_factory=lambda: deque(maxlen=200))

    # Derived metrics (updated after each observation)
    posterior_win_rate: float = 0.5
    posterior_variance: float = 0.25  # Var(Beta(1,1)) = 0.25
    rolling_ic: float = 0.0
    expected_edge: float = 0.0
    n_observations: int = 0

    @property
    def posterior_mean(self) -> float:
        """E[p] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        """Std(p) for Beta distribution."""
        a, b = self.alpha, self.beta
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def credible_interval_95(self) -> tuple[float, float]:
        """95% credible interval using normal approximation."""
        mean = self.posterior_mean
        std = self.posterior_std
        return (max(mean - 1.96 * std, 0.0), min(mean + 1.96 * std, 1.0))


class BayesianUpdater:
    """Bayesian updater for strategy weights.

    Maintains a Beta posterior per strategy for win rates,
    tracks rolling IC, and produces posterior-optimal weights
    for the AlphaCombiner.

    Parameters:
        prior_alpha: Initial Beta alpha (higher = stronger prior toward winning).
        prior_beta: Initial Beta beta.
        ic_window: Rolling window size for IC computation.
        min_observations: Minimum trades before posterior is trusted.
        decay_factor: Exponential decay for older observations (0-1, 1=no decay).
        regime_sensitivity: How quickly to detect regime changes.
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        ic_window: int = 100,
        min_observations: int = 10,
        decay_factor: float = 0.995,
        regime_sensitivity: float = 2.0,
    ) -> None:
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._ic_window = ic_window
        self._min_obs = min_observations
        self._decay = decay_factor
        self._regime_sensitivity = regime_sensitivity

        self._strategies: dict[str, StrategyPosterior] = {}
        self._global_observations: int = 0

    def register_strategy(
        self,
        name: str,
        prior_alpha: float | None = None,
        prior_beta: float | None = None,
    ) -> None:
        """Register a strategy with optional custom prior.

        An informative prior can encode domain knowledge, e.g.,
        Beta(8, 2) for a strategy expected to win ~80% of the time.
        """
        if name in self._strategies:
            return

        self._strategies[name] = StrategyPosterior(
            name=name,
            alpha=prior_alpha or self._prior_alpha,
            beta=prior_beta or self._prior_beta,
        )
        logger.info(
            "Registered strategy %s with prior Beta(%.1f, %.1f)",
            name,
            self._strategies[name].alpha,
            self._strategies[name].beta,
        )

    def record_prediction(
        self,
        strategy_name: str,
        predicted_edge: float,
        direction: str,
    ) -> None:
        """Record a prediction before the outcome is known.

        Used for IC computation (correlation of predictions vs outcomes).
        """
        if strategy_name not in self._strategies:
            self.register_strategy(strategy_name)

        signed_pred = predicted_edge if direction == "BUY" else -predicted_edge
        self._strategies[strategy_name].predictions.append(signed_pred)

    def update(
        self,
        strategy_name: str,
        won: bool,
        actual_return: float = 0.0,
    ) -> StrategyPosterior:
        """Update posterior after a trade resolves.

        Args:
            strategy_name: Which strategy generated the trade.
            won: Whether the trade was profitable.
            actual_return: Realized P&L (as fraction, e.g., 0.05 = +5%).

        Returns:
            Updated StrategyPosterior.
        """
        if strategy_name not in self._strategies:
            self.register_strategy(strategy_name)

        post = self._strategies[strategy_name]
        post.n_observations += 1
        self._global_observations += 1

        # --- Beta update with optional decay ---
        # Apply decay to prior counts to make posterior responsive to
        # regime changes (otherwise very old observations dominate)
        if self._decay < 1.0:
            post.alpha *= self._decay
            post.beta *= self._decay
            # Floor to prevent prior from vanishing entirely
            post.alpha = max(post.alpha, self._prior_alpha * 0.5)
            post.beta = max(post.beta, self._prior_beta * 0.5)

        # Conjugate update: observe success or failure
        if won:
            post.alpha += 1.0
        else:
            post.beta += 1.0

        # Update derived metrics
        post.posterior_win_rate = post.posterior_mean
        post.posterior_variance = post.posterior_std ** 2
        post.outcomes.append(actual_return)
        post.returns.append(actual_return)

        # Update rolling IC
        if len(post.predictions) >= self._min_obs and len(post.outcomes) >= self._min_obs:
            post.rolling_ic = self._compute_rolling_ic(post)

        # Expected edge = mean return over recent window
        if len(post.returns) >= self._min_obs:
            post.expected_edge = float(np.mean(list(post.returns)))

        logger.debug(
            "Updated %s: Beta(%.2f, %.2f) → WR=%.3f, IC=%.3f",
            strategy_name,
            post.alpha,
            post.beta,
            post.posterior_win_rate,
            post.rolling_ic,
        )

        return post

    def get_optimal_weights(self) -> dict[str, float]:
        """Compute posterior-optimal weights for all strategies.

        Weight is proportional to:
            w_i ∝ E[win_rate_i] × IC_i / σ_i

        This combines Bayesian win-rate estimation with signal quality
        (IC) and consistency (inverse volatility).
        """
        if not self._strategies:
            return {}

        raw_weights: dict[str, float] = {}

        for name, post in self._strategies.items():
            if post.n_observations < self._min_obs:
                # Not enough data — use prior mean with low weight
                raw_weights[name] = post.posterior_mean * 0.1
                continue

            # Win rate from posterior
            wr = post.posterior_win_rate

            # IC component (can be negative — disables the strategy)
            ic = max(post.rolling_ic, 0.0)

            # Volatility of returns
            if len(post.returns) >= 5:
                vol = float(np.std(list(post.returns)))
                vol = max(vol, 0.01)
            else:
                vol = 0.10  # default assumption

            # Composite score: Bayesian win rate * IC / vol
            # This is analogous to expected Sharpe ratio
            score = wr * (0.3 + 0.7 * ic) / vol

            raw_weights[name] = max(score, 0.0)

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total <= 0:
            n = len(self._strategies)
            return {name: 1.0 / n for name in self._strategies}

        return {name: w / total for name, w in raw_weights.items()}

    def apply_to_combiner(self, combiner: Any) -> None:
        """Push posterior weights into an AlphaCombiner instance.

        This is the bridge between Bayesian learning and the
        signal combination framework.
        """
        weights = self.get_optimal_weights()

        for name, weight in weights.items():
            if name in combiner._strategies:
                combiner._strategies[name].optimal_weight = weight

        logger.info(
            "Applied Bayesian weights to combiner: %s",
            {k: round(v, 4) for k, v in weights.items()},
        )

    def detect_regime_change(self, strategy_name: str) -> bool:
        """Detect if a strategy's IC has diverged from its prior.

        A regime change is flagged when recent IC is more than
        `regime_sensitivity` standard deviations from the long-run IC.
        """
        if strategy_name not in self._strategies:
            return False

        post = self._strategies[strategy_name]
        if post.n_observations < self._min_obs * 2:
            return False

        # Split returns into halves
        returns_list = list(post.returns)
        mid = len(returns_list) // 2
        first_half = np.array(returns_list[:mid])
        second_half = np.array(returns_list[mid:])

        if len(first_half) < 5 or len(second_half) < 5:
            return False

        mean_first = float(np.mean(first_half))
        mean_second = float(np.mean(second_half))
        std_first = float(np.std(first_half))

        if std_first < 1e-8:
            return False

        z_score = abs(mean_second - mean_first) / std_first
        return z_score > self._regime_sensitivity

    def get_strategy_summary(self) -> dict[str, dict[str, Any]]:
        """Return summary statistics for all tracked strategies."""
        summary = {}
        weights = self.get_optimal_weights()

        for name, post in self._strategies.items():
            ci_low, ci_high = post.credible_interval_95
            summary[name] = {
                "posterior_win_rate": round(post.posterior_win_rate, 4),
                "posterior_std": round(post.posterior_std, 4),
                "credible_interval_95": (round(ci_low, 4), round(ci_high, 4)),
                "rolling_ic": round(post.rolling_ic, 4),
                "expected_edge": round(post.expected_edge, 4),
                "n_observations": post.n_observations,
                "beta_alpha": round(post.alpha, 2),
                "beta_beta": round(post.beta, 2),
                "optimal_weight": round(weights.get(name, 0.0), 4),
                "regime_change": self.detect_regime_change(name),
            }

        return summary

    def reset_strategy(self, strategy_name: str) -> None:
        """Reset a strategy's posterior to the prior (e.g., after regime change)."""
        if strategy_name in self._strategies:
            post = self._strategies[strategy_name]
            post.alpha = self._prior_alpha
            post.beta = self._prior_beta
            post.predictions.clear()
            post.outcomes.clear()
            post.returns.clear()
            post.rolling_ic = 0.0
            post.expected_edge = 0.0
            post.n_observations = 0
            logger.info("Reset %s posterior to prior Beta(%.1f, %.1f)",
                        strategy_name, post.alpha, post.beta)

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _compute_rolling_ic(self, post: StrategyPosterior) -> float:
        """Compute IC as Spearman rank correlation of predictions vs outcomes."""
        preds = list(post.predictions)
        outs = list(post.outcomes)
        n = min(len(preds), len(outs), self._ic_window)

        if n < self._min_obs:
            return 0.0

        preds_arr = np.array(preds[-n:])
        outs_arr = np.array(outs[-n:])

        if np.std(preds_arr) < 1e-10 or np.std(outs_arr) < 1e-10:
            return 0.0

        # Spearman rank correlation
        pred_ranks = np.argsort(np.argsort(preds_arr)).astype(float)
        out_ranks = np.argsort(np.argsort(outs_arr)).astype(float)

        corr = np.corrcoef(pred_ranks, out_ranks)[0, 1]
        return float(np.nan_to_num(corr, 0.0))
