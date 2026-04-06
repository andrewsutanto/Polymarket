"""Order placement for paper and live trading modes.

Default is paper trading. Live mode requires three independent
activation flags to prevent accidental deployment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any

from config import settings
from core.risk_manager import TradeProposal

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    timestamp: datetime
    contract_id: str
    location: str
    target_date: str
    bucket_label: str
    side: str
    size_usd: float
    price: float
    edge: float
    confidence: float
    model_prob: float
    market_price: float
    forecast_high_f: float
    kelly_fraction: float
    mode: str  # PAPER / LIVE
    status: str  # FILLED / PARTIAL / CANCELLED
    notes: str


def _check_live_mode(cli_flag: bool) -> bool:
    """Verify all three independent live-trading flags are set."""
    flag_1 = cli_flag
    flag_2 = os.getenv("ENABLE_LIVE_TRADING", "").lower() == "true"
    flag_3 = settings.LIVE_TRADING_ENABLED
    return flag_1 and flag_2 and flag_3


class Executor:
    """Executes trades in paper or live mode."""

    def __init__(self, live_cli_flag: bool = False) -> None:
        self._live = _check_live_mode(live_cli_flag)
        self._mode = "LIVE" if self._live else "PAPER"
        self._callbacks: list[Callable[[TradeResult], Any]] = []
        self._clob_client: Any = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._live

    async def start(self) -> None:
        if self._live:
            await self._live_startup_sequence()

    async def stop(self) -> None:
        if self._clob_client:
            logger.info("Live executor shut down")

    def on_trade(self, cb: Callable[[TradeResult], Any]) -> None:
        self._callbacks.append(cb)

    async def execute(self, proposal: TradeProposal) -> TradeResult | None:
        """Execute a trade proposal.

        Args:
            proposal: An approved TradeProposal from the risk manager.

        Returns:
            TradeResult or None if execution fails.
        """
        if not proposal.approved:
            return None

        if self._live:
            return await self._execute_live(proposal)
        return await self._execute_paper(proposal)

    # ------------------------------------------------------------------
    # Paper trading
    # ------------------------------------------------------------------

    async def _execute_paper(self, proposal: TradeProposal) -> TradeResult:
        sig = proposal.signal
        price = sig.market_price

        # Simulate slippage
        if sig.signal_type == "BUY":
            price *= (1.0 + settings.SIMULATED_SLIPPAGE)
        else:
            price *= (1.0 - settings.SIMULATED_SLIPPAGE)

        result = TradeResult(
            timestamp=datetime.now(timezone.utc),
            contract_id=sig.contract_id,
            location=sig.location,
            target_date=str(sig.target_date),
            bucket_label=sig.bucket_label,
            side=sig.signal_type,
            size_usd=proposal.size_usd,
            price=round(price, 4),
            edge=sig.edge,
            confidence=sig.confidence,
            model_prob=sig.model_prob,
            market_price=sig.market_price,
            forecast_high_f=sig.forecast_high_f,
            kelly_fraction=proposal.kelly_fraction,
            mode="PAPER",
            status="FILLED",
            notes=f"Paper fill at simulated price {price:.4f}",
        )

        logger.info(
            "[PAPER] %s %s %s @ $%.4f | size: $%.2f | edge: %.0f%% | NOAA: %.0f°F",
            sig.signal_type, sig.location, sig.bucket_label, price,
            proposal.size_usd, sig.edge * 100, sig.forecast_high_f,
        )

        self._emit(result)
        return result

    # ------------------------------------------------------------------
    # Live trading
    # ------------------------------------------------------------------

    async def _execute_live(self, proposal: TradeProposal) -> TradeResult | None:
        sig = proposal.signal
        try:
            if not self._clob_client:
                await self._init_clob_client()

            if sig.signal_type == "BUY":
                order_price = sig.market_price
                result_status = await self._place_limit_order(
                    sig.contract_id, "BUY", proposal.size_usd, order_price
                )
            else:
                order_price = sig.market_price
                result_status = await self._place_limit_order(
                    sig.contract_id, "SELL", proposal.size_usd, order_price
                )

            result = TradeResult(
                timestamp=datetime.now(timezone.utc),
                contract_id=sig.contract_id,
                location=sig.location,
                target_date=str(sig.target_date),
                bucket_label=sig.bucket_label,
                side=sig.signal_type,
                size_usd=proposal.size_usd,
                price=round(order_price, 4),
                edge=sig.edge,
                confidence=sig.confidence,
                model_prob=sig.model_prob,
                market_price=sig.market_price,
                forecast_high_f=sig.forecast_high_f,
                kelly_fraction=proposal.kelly_fraction,
                mode="LIVE",
                status=result_status,
                notes="",
            )

            logger.info(
                "[LIVE] %s %s %s @ $%.4f | size: $%.2f | status: %s",
                sig.signal_type, sig.location, sig.bucket_label,
                order_price, proposal.size_usd, result_status,
            )

            self._emit(result)
            return result

        except Exception as exc:
            logger.error("Live execution failed: %s", exc)
            return None

    async def _init_clob_client(self) -> None:
        """Initialize the py-clob-client for live trading."""
        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                settings.POLYMARKET_HOST,
                key=settings.POLYMARKET_PRIVATE_KEY,
                chain_id=settings.POLYMARKET_CHAIN_ID,
            )
            logger.info("CLOB client initialized for live trading")
        except ImportError:
            logger.error("py-clob-client not installed — cannot trade live")
            self._live = False
            self._mode = "PAPER"

    async def _place_limit_order(
        self, token_id: str, side: str, size_usd: float, price: float
    ) -> str:
        """Place a limit order and wait for fill or timeout."""
        if not self._clob_client:
            return "CANCELLED"

        try:
            from py_clob_client.order_builder.constants import BUY as CLOB_BUY, SELL as CLOB_SELL

            clob_side = CLOB_BUY if side == "BUY" else CLOB_SELL
            order_args = {
                "token_id": token_id,
                "price": price,
                "size": size_usd / price if price > 0 else 0,
                "side": clob_side,
            }

            signed = self._clob_client.create_and_sign_order(order_args)
            resp = self._clob_client.post_order(signed)
            order_id = resp.get("orderID", "")

            # Wait for fill or timeout
            for _ in range(settings.ORDER_TIMEOUT):
                await asyncio.sleep(1)
                order_status = self._clob_client.get_order(order_id)
                status = order_status.get("status", "")
                if status == "FILLED":
                    return "FILLED"
                if status == "PARTIAL":
                    return "PARTIAL"

            # Timeout — cancel
            self._clob_client.cancel(order_id)
            return "CANCELLED"

        except Exception as exc:
            logger.error("Order placement error: %s", exc)
            return "CANCELLED"

    # ------------------------------------------------------------------
    # Live startup safety
    # ------------------------------------------------------------------

    async def _live_startup_sequence(self) -> None:
        """Print warning and require confirmation countdown for live mode."""
        print("\n" + "=" * 60)
        print("  WARNING: LIVE TRADING MODE ENABLED")
        print("  Real money will be used for trades!")
        print("=" * 60)
        print("\nStarting in 10 seconds... Press Ctrl+C to abort.\n")

        for i in range(10, 0, -1):
            print(f"  {i}...", flush=True)
            await asyncio.sleep(1)

        print("\n  LIVE TRADING ACTIVE\n")
        logger.warning("Live trading mode confirmed and active")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _emit(self, result: TradeResult) -> None:
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception:
                logger.exception("Trade callback error")
