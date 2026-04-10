"""Order placement for paper and live trading modes.

Default is paper trading. Live mode requires three independent
activation flags to prevent accidental deployment.

Live execution features:
- Fill confirmation polling with configurable timeout
- Partial fill tracking with remainder cancellation
- Order state machine: PENDING -> PARTIAL/FILLED/CANCELLED/EXPIRED
- Nonce tracking to prevent duplicate on-chain transactions
- Exponential backoff on poll failures
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any

from config import settings
from core.risk_manager import TradeProposal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Order state machine
# ------------------------------------------------------------------


class OrderState(enum.Enum):
    """Lifecycle states for a live order."""

    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderRecord:
    """Tracks the full lifecycle of a single live order."""

    order_id: str
    token_id: str
    side: str
    requested_size: float  # shares requested
    requested_price: float
    size_usd: float
    state: OrderState = OrderState.PENDING
    filled_size: float = 0.0
    filled_avg_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancel_attempted: bool = False
    nonce: int = 0


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
    filled_size: float = 0.0  # actual shares filled (live only)
    filled_usd: float = 0.0  # actual USD filled (live only)
    order_id: str = ""  # CLOB order ID (live only)


def _check_live_mode(cli_flag: bool) -> bool:
    """Verify all three independent live-trading flags are set."""
    flag_1 = cli_flag
    flag_2 = os.getenv("ENABLE_LIVE_TRADING", "").lower() == "true"
    flag_3 = settings.LIVE_TRADING_ENABLED
    return flag_1 and flag_2 and flag_3


# ------------------------------------------------------------------
# Nonce tracker
# ------------------------------------------------------------------


class NonceTracker:
    """Thread-safe monotonic nonce for on-chain transaction ordering.

    Prevents duplicate orders by ensuring each signed transaction
    uses a strictly increasing nonce. Persists the high-water mark
    in memory; on restart it re-syncs from the chain via the CLOB
    client if available.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nonce: int = 0
        self._initialized = False

    def initialize(self, clob_client: Any) -> None:
        """Sync nonce from the on-chain state."""
        with self._lock:
            if self._initialized:
                return
            try:
                # py_clob_client exposes get_nonce() for the signing wallet
                if hasattr(clob_client, "get_nonce"):
                    chain_nonce = clob_client.get_nonce()
                    self._nonce = int(chain_nonce)
                    logger.info("Nonce synced from chain: %d", self._nonce)
                else:
                    # Fallback: use millisecond timestamp as starting nonce
                    self._nonce = int(time.time() * 1000)
                    logger.info("Nonce initialized from timestamp: %d", self._nonce)
            except Exception as exc:
                self._nonce = int(time.time() * 1000)
                logger.warning("Nonce sync failed (%s), using timestamp: %d", exc, self._nonce)
            self._initialized = True

    def next(self) -> int:
        """Return the next nonce and increment."""
        with self._lock:
            if not self._initialized:
                self._nonce = int(time.time() * 1000)
                self._initialized = True
            nonce = self._nonce
            self._nonce += 1
            return nonce

    @property
    def current(self) -> int:
        with self._lock:
            return self._nonce


# ------------------------------------------------------------------
# Poll configuration
# ------------------------------------------------------------------

# Exponential backoff defaults (overridden by settings if present)
_POLL_BASE_INTERVAL: float = getattr(settings, "ORDER_POLL_INTERVAL", 1.0)
_POLL_MAX_INTERVAL: float = getattr(settings, "ORDER_POLL_MAX_INTERVAL", 5.0)
_POLL_BACKOFF_FACTOR: float = getattr(settings, "ORDER_POLL_BACKOFF", 1.5)
_POLL_MAX_CONSECUTIVE_ERRORS: int = getattr(settings, "ORDER_MAX_POLL_ERRORS", 5)


class Executor:
    """Executes trades in paper or live mode."""

    def __init__(self, live_cli_flag: bool = False) -> None:
        self._live = _check_live_mode(live_cli_flag)
        self._mode = "LIVE" if self._live else "PAPER"
        self._callbacks: list[Callable[[TradeResult], Any]] = []
        self._clob_client: Any = None
        # Order state tracking: order_id -> OrderRecord
        self._orders: dict[str, OrderRecord] = {}
        self._nonce_tracker = NonceTracker()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def orders(self) -> dict[str, OrderRecord]:
        """Read-only view of all tracked orders."""
        return dict(self._orders)

    def get_order(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[OrderRecord]:
        """Return orders that are still PENDING or PARTIAL."""
        return [
            o for o in self._orders.values()
            if o.state in (OrderState.PENDING, OrderState.PARTIAL)
        ]

    async def start(self) -> None:
        if self._live:
            await self._live_startup_sequence()

    async def stop(self) -> None:
        """Shut down: cancel all open orders, then close client."""
        if self._clob_client:
            open_orders = self.get_open_orders()
            if open_orders:
                logger.info("Cancelling %d open orders on shutdown", len(open_orders))
                for rec in open_orders:
                    await self._cancel_order(rec)
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
            "[PAPER] %s %s %s @ $%.4f | size: $%.2f | edge: %.0f%% | NOAA: %.0f\u00b0F",
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

            order_price = sig.market_price
            fill_result = await self._place_and_confirm_order(
                token_id=sig.contract_id,
                side=sig.signal_type,
                size_usd=proposal.size_usd,
                price=order_price,
            )

            result = TradeResult(
                timestamp=datetime.now(timezone.utc),
                contract_id=sig.contract_id,
                location=sig.location,
                target_date=str(sig.target_date),
                bucket_label=sig.bucket_label,
                side=sig.signal_type,
                size_usd=proposal.size_usd,
                price=round(fill_result.filled_avg_price or order_price, 4),
                edge=sig.edge,
                confidence=sig.confidence,
                model_prob=sig.model_prob,
                market_price=sig.market_price,
                forecast_high_f=sig.forecast_high_f,
                kelly_fraction=proposal.kelly_fraction,
                mode="LIVE",
                status=fill_result.state.value,
                notes=self._build_fill_notes(fill_result),
                filled_size=fill_result.filled_size,
                filled_usd=fill_result.filled_size * fill_result.filled_avg_price
                if fill_result.filled_avg_price > 0 else 0.0,
                order_id=fill_result.order_id,
            )

            logger.info(
                "[LIVE] %s %s %s @ $%.4f | size: $%.2f | status: %s | "
                "filled: %.4f shares @ avg $%.4f | order: %s",
                sig.signal_type, sig.location, sig.bucket_label,
                order_price, proposal.size_usd, fill_result.state.value,
                fill_result.filled_size, fill_result.filled_avg_price,
                fill_result.order_id,
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
                signature_type=2,  # EIP-1559 for Polygon
            )

            # Set API credentials if available
            if settings.POLYMARKET_API_KEY:
                self._clob_client.set_api_creds(
                    self._clob_client.create_or_derive_api_creds()
                )

            # Sync nonce from chain
            self._nonce_tracker.initialize(self._clob_client)

            logger.info("CLOB client initialized for live trading")
        except ImportError:
            logger.error("py-clob-client not installed — cannot trade live")
            self._live = False
            self._mode = "PAPER"

    async def _place_and_confirm_order(
        self, token_id: str, side: str, size_usd: float, price: float
    ) -> OrderRecord:
        """Place a limit order, poll for fill confirmation, handle partials.

        Returns an OrderRecord with final state after polling completes.
        On timeout, cancels the unfilled remainder.
        """
        if not self._clob_client:
            return OrderRecord(
                order_id="",
                token_id=token_id,
                side=side,
                requested_size=0,
                requested_price=price,
                size_usd=size_usd,
                state=OrderState.CANCELLED,
            )

        try:
            from py_clob_client.order_builder.constants import BUY as CLOB_BUY, SELL as CLOB_SELL

            clob_side = CLOB_BUY if side == "BUY" else CLOB_SELL
            requested_size = size_usd / price if price > 0 else 0

            # Acquire nonce for this order
            nonce = self._nonce_tracker.next()

            order_args = {
                "token_id": token_id,
                "price": price,
                "size": requested_size,
                "side": clob_side,
                "nonce": nonce,
            }

            # Sign and submit
            signed = self._clob_client.create_and_sign_order(order_args)
            resp = self._clob_client.post_order(signed)
            order_id = resp.get("orderID", "")

            if not order_id:
                error_msg = resp.get("errorMsg", resp.get("error", "no orderID returned"))
                logger.error("Order submission failed: %s", error_msg)
                return OrderRecord(
                    order_id="",
                    token_id=token_id,
                    side=side,
                    requested_size=requested_size,
                    requested_price=price,
                    size_usd=size_usd,
                    state=OrderState.CANCELLED,
                    nonce=nonce,
                )

            # Create tracking record
            record = OrderRecord(
                order_id=order_id,
                token_id=token_id,
                side=side,
                requested_size=requested_size,
                requested_price=price,
                size_usd=size_usd,
                state=OrderState.PENDING,
                nonce=nonce,
            )
            self._orders[order_id] = record

            logger.info(
                "Order placed: %s %s %.4f shares @ $%.4f | id=%s nonce=%d",
                side, token_id, requested_size, price, order_id, nonce,
            )

            # Poll for fill confirmation
            record = await self._poll_fill_status(record)

            # Handle timeout: cancel unfilled remainder
            if record.state == OrderState.PENDING:
                # Nothing filled within timeout -- cancel and mark EXPIRED
                await self._cancel_order(record)
                record.state = OrderState.EXPIRED
                record.updated_at = datetime.now(timezone.utc)
                logger.warning(
                    "Order expired (no fills in %ds): %s", settings.ORDER_TIMEOUT, order_id
                )

            elif record.state == OrderState.PARTIAL:
                # Partially filled -- cancel the remainder
                await self._cancel_order(record)
                logger.info(
                    "Partial fill: %.4f/%.4f shares filled, remainder cancelled: %s",
                    record.filled_size, record.requested_size, order_id,
                )

            self._orders[order_id] = record
            return record

        except Exception as exc:
            logger.error("Order placement error: %s", exc)
            return OrderRecord(
                order_id="",
                token_id=token_id,
                side=side,
                requested_size=size_usd / price if price > 0 else 0,
                requested_price=price,
                size_usd=size_usd,
                state=OrderState.CANCELLED,
            )

    async def _poll_fill_status(self, record: OrderRecord) -> OrderRecord:
        """Poll the CLOB API for order fill status with exponential backoff.

        Polls until:
        - Order is fully filled (FILLED)
        - Timeout exceeded (returns PENDING or PARTIAL for caller to handle)
        - Too many consecutive API errors

        Returns the updated OrderRecord.
        """
        timeout_s = settings.ORDER_TIMEOUT
        deadline = time.monotonic() + timeout_s
        poll_interval = _POLL_BASE_INTERVAL
        consecutive_errors = 0

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)

            try:
                order_data = await asyncio.to_thread(
                    self._clob_client.get_order, record.order_id
                )
                consecutive_errors = 0  # reset on success
            except Exception as exc:
                consecutive_errors += 1
                logger.debug(
                    "Poll error for %s (attempt %d): %s",
                    record.order_id, consecutive_errors, exc,
                )
                if consecutive_errors >= _POLL_MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "Too many poll errors for %s, aborting poll", record.order_id
                    )
                    break
                # Backoff faster on errors
                poll_interval = min(poll_interval * _POLL_BACKOFF_FACTOR, _POLL_MAX_INTERVAL)
                continue

            # Parse fill data from CLOB response
            status = order_data.get("status", "").upper()
            size_matched = float(order_data.get("size_matched", 0))
            avg_price = float(order_data.get("associate_trades", {}).get("avg_price", 0))

            # Fallback: compute avg_price from matched size if not provided
            if avg_price == 0 and size_matched > 0:
                avg_price = record.requested_price  # best estimate

            record.filled_size = size_matched
            record.filled_avg_price = avg_price
            record.updated_at = datetime.now(timezone.utc)

            if status == "MATCHED" or status == "FILLED":
                record.state = OrderState.FILLED
                record.filled_size = size_matched or record.requested_size
                logger.info(
                    "Order fully filled: %s | %.4f shares @ avg $%.4f",
                    record.order_id, record.filled_size, record.filled_avg_price,
                )
                return record

            if status == "CANCELLED" or status == "CANCELED":
                record.state = OrderState.CANCELLED
                logger.info("Order was cancelled externally: %s", record.order_id)
                return record

            # Check for partial fills
            if size_matched > 0 and size_matched < record.requested_size:
                record.state = OrderState.PARTIAL
                logger.debug(
                    "Partial fill in progress: %s | %.4f/%.4f",
                    record.order_id, size_matched, record.requested_size,
                )
                # Use shorter interval when partially filled (fills may come faster)
                poll_interval = _POLL_BASE_INTERVAL
            else:
                # Normal backoff for pending orders
                poll_interval = min(poll_interval * _POLL_BACKOFF_FACTOR, _POLL_MAX_INTERVAL)

        return record

    async def _cancel_order(self, record: OrderRecord) -> bool:
        """Cancel an order via the CLOB API. Returns True on success."""
        if record.cancel_attempted:
            return False
        if not record.order_id:
            return False

        record.cancel_attempted = True

        try:
            resp = await asyncio.to_thread(
                self._clob_client.cancel, record.order_id
            )
            cancelled = resp.get("canceled", False) if isinstance(resp, dict) else bool(resp)

            if cancelled:
                logger.info("Order cancelled: %s", record.order_id)
                # Only set CANCELLED if nothing was filled
                if record.filled_size == 0:
                    record.state = OrderState.CANCELLED
                # If partial, keep PARTIAL state -- caller decides
                record.updated_at = datetime.now(timezone.utc)
                return True
            else:
                logger.warning(
                    "Cancel request returned non-success for %s: %s",
                    record.order_id, resp,
                )
                return False

        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", record.order_id, exc)
            return False

    async def cancel_all_open_orders(self) -> int:
        """Cancel all open orders. Returns the count of orders cancelled."""
        cancelled = 0
        for rec in self.get_open_orders():
            if await self._cancel_order(rec):
                cancelled += 1
        return cancelled

    @staticmethod
    def _build_fill_notes(record: OrderRecord) -> str:
        """Build human-readable notes for a fill result."""
        if record.state == OrderState.FILLED:
            return (
                f"Fully filled: {record.filled_size:.4f} shares "
                f"@ avg ${record.filled_avg_price:.4f}"
            )
        if record.state == OrderState.PARTIAL:
            fill_pct = (
                (record.filled_size / record.requested_size * 100)
                if record.requested_size > 0 else 0
            )
            return (
                f"Partial fill: {record.filled_size:.4f}/{record.requested_size:.4f} "
                f"shares ({fill_pct:.1f}%) @ avg ${record.filled_avg_price:.4f}, "
                f"remainder cancelled"
            )
        if record.state == OrderState.EXPIRED:
            return f"Expired: no fills within {settings.ORDER_TIMEOUT}s timeout"
        if record.state == OrderState.CANCELLED:
            return "Cancelled before any fill"
        return ""

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
