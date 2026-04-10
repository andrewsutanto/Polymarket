"""Wallet Copy-Trade strategy — BaseStrategy wrapper for copy trading.

Generates signals when profitable watched wallets trade a market.
The edge is derived from the wallet's historical performance,
and the signal strength is proportional to the wallet's score.

This is a fundamentally different alpha source from the other strategies:
instead of predicting price from market features, we follow wallets
with proven track records. The information advantage comes from
the public nature of Polygon blockchain transactions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.wallet_tracker import (
    CopySignal,
    WalletCopyTrader,
    WalletScoreDB,
)
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class WalletCopyTradeStrategy(BaseStrategy):
    """BaseStrategy wrapper that emits signals from wallet copy-trading.

    During generate_signal(), checks if any watched wallet recently traded
    the given market. If so, returns a Signal with:
        - direction: same as the wallet's trade
        - edge: wallet's historical edge * confidence factor
        - strength: wallet score normalized to 0-1

    Parameters:
        min_edge: Minimum estimated edge to emit a signal (default 2%)
        min_wallet_score: Minimum wallet score to copy (default 1.0)
        confidence_factor: Scale factor for estimated edge (default 0.8)
        max_copy_delay_s: Maximum seconds since wallet trade (default 60)
        max_price_move: Maximum price move since wallet trade (default 2%)
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        min_wallet_score: float = 1.0,
        confidence_factor: float = 0.8,
        max_copy_delay_s: float = 60.0,
        max_price_move: float = 0.02,
        db_path: str = "data/wallet_scores.db",
    ) -> None:
        self._min_edge = min_edge
        self._min_wallet_score = min_wallet_score
        self._confidence_factor = confidence_factor
        self._max_copy_delay_s = max_copy_delay_s
        self._max_price_move = max_price_move

        self._db = WalletScoreDB(db_path)
        self._copier = WalletCopyTrader(
            db=self._db,
            max_price_move=max_price_move,
            max_copy_delay_s=max_copy_delay_s,
            confidence_factor=confidence_factor,
        )

        # Pending signals from the copy trader, keyed by market_id
        # Set externally by the bot when new wallet trades are detected
        self._pending_signals: dict[str, CopySignal] = {}

    @property
    def name(self) -> str:
        return "wallet_copytrade"

    @property
    def copier(self) -> WalletCopyTrader:
        """Access the underlying WalletCopyTrader for monitoring."""
        return self._copier

    def inject_signal(self, signal: CopySignal) -> None:
        """Inject a copy signal for a market (called by the bot scan loop).

        The bot monitors wallet trades and injects signals here.
        When generate_signal() is called for this market, it will pick it up.
        """
        self._pending_signals[signal.market_id] = signal

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Check if a watched wallet recently traded this market.

        If a pending copy signal exists for this market, validate it
        and return a Signal. Otherwise return None.
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")

        # Check for pending copy signal
        copy_sig = self._pending_signals.pop(market_id, None)
        if copy_sig is None:
            return None

        # Validate wallet score
        if copy_sig.wallet_score < self._min_wallet_score:
            logger.debug(
                "Skip copy: wallet score %.2f < min %.2f",
                copy_sig.wallet_score, self._min_wallet_score,
            )
            return None

        # Compute edge
        edge = self._copier.compute_edge(copy_sig)
        if edge < self._min_edge:
            logger.debug(
                "Skip copy: edge %.3f < min %.3f",
                edge, self._min_edge,
            )
            return None

        # Anti-front-run: check current price hasn't moved too far
        current_price = market_data.get("mid_price", market_data.get("yes_price", 0))
        if current_price > 0 and copy_sig.wallet_price > 0:
            move = abs(current_price - copy_sig.wallet_price) / copy_sig.wallet_price
            if move > self._max_price_move:
                logger.info(
                    "Skip copy: price moved %.1f%% since wallet trade",
                    move * 100,
                )
                return None

        strength = self._copier.get_signal_strength(copy_sig)

        return Signal(
            direction=copy_sig.direction,
            strength=strength,
            edge=edge,
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id or copy_sig.token_id,
            market_slug=market_data.get("market_slug", ""),
            category=market_data.get("category", "other"),
            outcome=copy_sig.outcome or market_data.get("outcome", ""),
            metadata={
                "wallet": copy_sig.wallet,
                "wallet_score": copy_sig.wallet_score,
                "wallet_price": copy_sig.wallet_price,
                "copy_delay_s": copy_sig.delay_s,
                "current_price": current_price,
            },
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_edge": self._min_edge,
            "min_wallet_score": self._min_wallet_score,
            "confidence_factor": self._confidence_factor,
            "max_copy_delay_s": self._max_copy_delay_s,
            "max_price_move": self._max_price_move,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        mapping = {
            "min_edge": "_min_edge",
            "min_wallet_score": "_min_wallet_score",
            "confidence_factor": "_confidence_factor",
            "max_copy_delay_s": "_max_copy_delay_s",
            "max_price_move": "_max_price_move",
        }
        for k, v in params.items():
            attr = mapping.get(k)
            if attr and hasattr(self, attr):
                setattr(self, attr, v)

    def reset(self) -> None:
        """Reset internal state between scans."""
        self._pending_signals.clear()
        self._copier.cleanup_stale_copies()
