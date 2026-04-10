"""Markov Chain pricing model for prediction markets.

Discretizes price history into states, builds a transition matrix,
and runs Monte Carlo simulations to estimate true resolution probability.

Based on the quant framework: build transition matrix → simulate 10K futures
→ calibrate against longshot bias → compare to market price → find edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarkovEstimate:
    """Result of a Markov Chain Monte Carlo probability estimate."""

    raw_probability: float  # Raw MC estimate before calibration
    calibrated_probability: float  # After longshot bias adjustment
    market_price: float
    edge: float  # calibrated_prob - market_price
    confidence: float  # Based on data quality and convergence
    n_simulations: int
    n_history_points: int
    steady_state: np.ndarray | None  # Steady-state distribution
    metadata: dict[str, Any]


class MarkovModel:
    """Markov Chain model for prediction market pricing.

    Discretizes price into N states, builds transition matrix from
    observed price history, and runs Monte Carlo to estimate true
    resolution probability.
    """

    def __init__(
        self,
        n_states: int = 10,
        n_simulations: int = 10_000,
        default_horizon_days: int = 30,
    ) -> None:
        self._n_states = n_states
        self._n_sims = n_simulations
        self._horizon = default_horizon_days

        # Cache transition matrices per market
        self._matrices: dict[str, np.ndarray] = {}
        self._history_lengths: dict[str, int] = {}

    def reset(self) -> None:
        """Clear all cached matrices and history. Call between sim runs."""
        self._matrices.clear()
        self._history_lengths.clear()

    @property
    def n_states(self) -> int:
        return self._n_states

    def build_transition_matrix(
        self, prices: list[float] | np.ndarray
    ) -> np.ndarray:
        """Build transition matrix from price history.

        Args:
            prices: Historical prices in [0, 1] range.

        Returns:
            n_states x n_states transition probability matrix.
        """
        prices = np.array(prices, dtype=float)
        prices = np.clip(prices, 0.001, 0.999)

        states = np.clip(
            (prices * self._n_states).astype(int), 0, self._n_states - 1
        )

        T = np.zeros((self._n_states, self._n_states))
        for i in range(len(states) - 1):
            T[states[i], states[i + 1]] += 1

        # Normalize rows to probabilities
        row_sums = T.sum(axis=1, keepdims=True)
        # For rows with no transitions, use uniform distribution
        zero_rows = row_sums.flatten() == 0
        T[zero_rows] = 1.0 / self._n_states
        row_sums[zero_rows] = 1.0
        T = T / np.where(row_sums > 0, row_sums, 1.0)

        return T

    def monte_carlo_probability(
        self,
        T: np.ndarray,
        start_price: float,
        horizon_steps: int | None = None,
        threshold: float = 0.5,
    ) -> float:
        """Run Monte Carlo simulation through transition matrix.

        Args:
            T: Transition matrix.
            start_price: Current market price [0, 1].
            horizon_steps: Number of steps to simulate forward.
            threshold: Price threshold for YES resolution.

        Returns:
            Estimated probability of resolving YES.
        """
        if horizon_steps is None:
            horizon_steps = self._horizon

        start_state = min(
            int(start_price * self._n_states), self._n_states - 1
        )
        start_state = max(0, start_state)

        # Vectorized simulation for speed
        states = np.full(self._n_sims, start_state, dtype=int)
        cumulative_probs = np.array([T[s] for s in range(self._n_states)])

        for _ in range(horizon_steps):
            # For each simulation, sample next state from transition probs
            randoms = np.random.random(self._n_sims)
            cum_probs = np.cumsum(T[states], axis=1)
            states = np.array([
                np.searchsorted(cum_probs[i], randoms[i])
                for i in range(self._n_sims)
            ])
            states = np.clip(states, 0, self._n_states - 1)

        # Count how many end above threshold
        threshold_state = int(threshold * self._n_states)
        p_yes = (states >= threshold_state).mean()
        return float(p_yes)

    def steady_state_distribution(self, T: np.ndarray) -> np.ndarray:
        """Compute steady-state distribution via eigenvalue decomposition.

        The steady state tells us the long-run probability of being
        in each price state — useful for identifying where the market
        "wants" to settle.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(T.T)
            # Find eigenvector for eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            steady = np.real(eigenvectors[:, idx])
            steady = steady / steady.sum()
            # Ensure non-negative
            steady = np.maximum(steady, 0)
            if steady.sum() > 0:
                steady = steady / steady.sum()
            return steady
        except np.linalg.LinAlgError:
            return np.ones(self._n_states) / self._n_states

    def absorbing_probability(self, T: np.ndarray, start_price: float) -> float:
        """Compute absorption probability for YES outcome.

        Models the market as having two absorbing states:
        state 0 (NO, price → 0) and state N-1 (YES, price → 1).

        Returns probability of being absorbed into YES state.
        """
        n = self._n_states
        if n < 3:
            return 0.5

        # Extract transient states (1 to n-2)
        Q = T[1:n-1, 1:n-1]  # Transient-to-transient
        R = np.zeros((n - 2, 2))
        R[:, 0] = T[1:n-1, 0]      # Transient-to-NO (absorbing)
        R[:, 1] = T[1:n-1, n-1]    # Transient-to-YES (absorbing)

        try:
            # Fundamental matrix N = (I - Q)^-1
            I = np.eye(n - 2)
            N = np.linalg.inv(I - Q)
            # Absorption probabilities B = N * R
            B = N @ R

            start_state = min(
                int(start_price * n), n - 1
            )
            start_state = max(1, min(start_state, n - 2))
            transient_idx = start_state - 1

            p_yes = float(B[transient_idx, 1])
            return np.clip(p_yes, 0.0, 1.0)
        except np.linalg.LinAlgError:
            return self.monte_carlo_probability(T, start_price)

    def estimate(
        self,
        market_id: str,
        prices: list[float] | np.ndarray,
        current_price: float,
        horizon_steps: int | None = None,
        calibrator: Any | None = None,
    ) -> MarkovEstimate:
        """Full estimation pipeline for a market.

        Args:
            market_id: Unique market identifier.
            prices: Historical price series.
            current_price: Current market price.
            horizon_steps: Forward simulation steps.
            calibrator: Optional BiasCalibrator for longshot adjustment.

        Returns:
            MarkovEstimate with raw and calibrated probabilities.
        """
        prices_arr = np.array(prices, dtype=float)
        if len(prices_arr) < 5:
            return MarkovEstimate(
                raw_probability=current_price,
                calibrated_probability=current_price,
                market_price=current_price,
                edge=0.0,
                confidence=0.0,
                n_simulations=0,
                n_history_points=len(prices_arr),
                steady_state=None,
                metadata={"reason": "insufficient_history"},
            )

        T = self.build_transition_matrix(prices_arr)
        self._matrices[market_id] = T
        self._history_lengths[market_id] = len(prices_arr)

        # Run Monte Carlo
        raw_prob = self.monte_carlo_probability(
            T, current_price, horizon_steps
        )

        # Also compute absorbing probability for comparison
        absorb_prob = self.absorbing_probability(T, current_price)

        # Blend MC and absorbing (MC more reliable with more data)
        data_quality = min(len(prices_arr) / 100.0, 1.0)
        blended = raw_prob * 0.6 + absorb_prob * 0.4

        # Calibrate against longshot bias
        if calibrator is not None:
            calibrated = calibrator.calibrate(blended, current_price)
        else:
            calibrated = blended

        # Confidence based on data quantity and convergence
        confidence = min(len(prices_arr) / 50.0, 1.0) * 0.7
        # Boost confidence if MC and absorbing agree
        agreement = 1.0 - abs(raw_prob - absorb_prob)
        confidence += agreement * 0.3

        steady = self.steady_state_distribution(T)
        edge = calibrated - current_price

        return MarkovEstimate(
            raw_probability=round(raw_prob, 4),
            calibrated_probability=round(calibrated, 4),
            market_price=round(current_price, 4),
            edge=round(edge, 4),
            confidence=round(min(confidence, 1.0), 4),
            n_simulations=self._n_sims,
            n_history_points=len(prices_arr),
            steady_state=steady,
            metadata={
                "absorbing_prob": round(absorb_prob, 4),
                "mc_prob": round(raw_prob, 4),
                "blended_prob": round(blended, 4),
                "data_quality": round(data_quality, 4),
                "transition_matrix_nonzero": int(np.count_nonzero(T)),
            },
        )

    def get_transition_matrix(self, market_id: str) -> np.ndarray | None:
        return self._matrices.get(market_id)
