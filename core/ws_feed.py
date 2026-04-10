"""Real-time WebSocket feed for Polymarket CLOB orderbook updates.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market for
streaming book snapshots and incremental price_change updates. Falls
back to REST polling when the WebSocket is disconnected.

Drop-in replacement for CLOBFeed — exposes the same interface:
    track_token / untrack_token / clear_tracked
    get_snapshot / get_all_snapshots
    start / stop
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from config import settings
from core.clob_feed import BookSnapshot

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL_S = 10          # Server expects PING every 10 s
RECONNECT_BASE_S = 1.0        # Exponential backoff base
RECONNECT_MAX_S = 60.0        # Cap on backoff
RECONNECT_FACTOR = 2.0
SUBSCRIBE_BATCH_SIZE = 200    # Max assets per subscription message
REST_FALLBACK_INTERVAL_S = 20 # Poll interval when WS is down


@dataclass
class _TokenMeta:
    """Metadata for a tracked token."""
    condition_id: str
    outcome: str


class WSFeed:
    """Real-time WebSocket feed with REST-polling fallback.

    Thread-safe: the scan loop in bot.py can call get_snapshot / get_all_snapshots
    from any asyncio task while the WS listener runs concurrently.
    """

    def __init__(self) -> None:
        # Token tracking  (token_id -> metadata)
        self._tracked: dict[str, _TokenMeta] = {}

        # In-memory orderbooks  (token_id -> BookSnapshot)
        # Protected by _lock for thread-safe reads from scan loop
        self._snapshots: dict[str, BookSnapshot] = {}
        self._lock = threading.Lock()

        # Internal state
        self._running = False
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ws_connected = False
        self._reconnect_delay = RECONNECT_BASE_S
        self._tasks: list[asyncio.Task] = []

        # Stats
        self._ws_messages_received = 0
        self._rest_polls = 0
        self._reconnect_count = 0

    # ── Public interface (same as CLOBFeed) ──────────────────────────

    @property
    def snapshot_count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    @property
    def is_ws_connected(self) -> bool:
        return self._ws_connected

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "ws_connected": self._ws_connected,
            "ws_messages": self._ws_messages_received,
            "rest_polls": self._rest_polls,
            "reconnects": self._reconnect_count,
            "tracked_tokens": len(self._tracked),
            "snapshots": self.snapshot_count,
        }

    async def start(self) -> None:
        """Start the WebSocket listener and fallback poller."""
        self._session = aiohttp.ClientSession()
        self._running = True

        # Launch WS connection loop and fallback poller as concurrent tasks
        ws_task = asyncio.create_task(self._ws_loop(), name="ws_feed_loop")
        fb_task = asyncio.create_task(self._fallback_poll_loop(), name="ws_fallback_poll")
        self._tasks = [ws_task, fb_task]
        logger.info("WSFeed started (WebSocket + REST fallback)")

    async def stop(self) -> None:
        """Gracefully shut down."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        # Wait for tasks to finish (suppress CancelledError)
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._ws_connected = False
        logger.info("WSFeed stopped")

    def track_token(self, token_id: str, condition_id: str, outcome: str) -> None:
        """Add a token to the tracking set.  Sends a live subscribe if WS is up."""
        self._tracked[token_id] = _TokenMeta(condition_id=condition_id, outcome=outcome)
        # Queue a dynamic subscribe (fire-and-forget)
        if self._ws_connected and self._ws and not self._ws.closed:
            asyncio.ensure_future(self._dynamic_subscribe([token_id]))

    def untrack_token(self, token_id: str) -> None:
        self._tracked.pop(token_id, None)
        with self._lock:
            self._snapshots.pop(token_id, None)

    def clear_tracked(self) -> None:
        self._tracked.clear()
        with self._lock:
            self._snapshots.clear()

    def get_snapshot(self, token_id: str) -> BookSnapshot | None:
        with self._lock:
            return self._snapshots.get(token_id)

    def get_all_snapshots(self) -> dict[str, BookSnapshot]:
        with self._lock:
            return dict(self._snapshots)

    # ── WebSocket lifecycle ──────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """Persistent reconnect loop around the WebSocket connection."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("WS connection lost: %s", e)

            self._ws_connected = False

            if not self._running:
                break

            # Exponential backoff before reconnect
            delay = min(self._reconnect_delay, RECONNECT_MAX_S)
            logger.info("WS reconnecting in %.1fs ...", delay)
            await asyncio.sleep(delay)
            self._reconnect_delay = min(
                self._reconnect_delay * RECONNECT_FACTOR, RECONNECT_MAX_S
            )
            self._reconnect_count += 1

    async def _connect_and_listen(self) -> None:
        """Open a single WS connection, subscribe, and process messages."""
        assert self._session is not None
        logger.info("Connecting to %s ...", WS_MARKET_URL)

        async with self._session.ws_connect(
            WS_MARKET_URL,
            heartbeat=PING_INTERVAL_S,
            timeout=aiohttp.ClientWSTimeout(ws_close=10),
        ) as ws:
            self._ws = ws
            self._ws_connected = True
            self._reconnect_delay = RECONNECT_BASE_S  # Reset backoff on success
            logger.info("WS connected")

            # Subscribe to all currently tracked tokens
            await self._subscribe_all()

            # Start ping task
            ping_task = asyncio.create_task(self._ping_loop(ws), name="ws_ping")

            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error("WS error: %s", ws.exception())
                        break
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        logger.info("WS closed by server")
                        break
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send PING every 10 s to keep the connection alive."""
        while not ws.closed:
            try:
                await ws.send_str("PING")
            except Exception:
                break
            await asyncio.sleep(PING_INTERVAL_S)

    # ── Subscription management ──────────────────────────────────────

    async def _subscribe_all(self) -> None:
        """Subscribe to all currently tracked token IDs."""
        token_ids = list(self._tracked.keys())
        if not token_ids:
            return

        # Batch into chunks of SUBSCRIBE_BATCH_SIZE
        for i in range(0, len(token_ids), SUBSCRIBE_BATCH_SIZE):
            batch = token_ids[i : i + SUBSCRIBE_BATCH_SIZE]
            msg = json.dumps({
                "assets_ids": batch,
                "type": "market",
                "custom_feature_enabled": True,
            })
            if self._ws and not self._ws.closed:
                await self._ws.send_str(msg)
                logger.debug("Subscribed to %d tokens", len(batch))

    async def _dynamic_subscribe(self, token_ids: list[str]) -> None:
        """Subscribe to new tokens on an existing connection."""
        if not token_ids or not self._ws or self._ws.closed:
            return
        msg = json.dumps({
            "assets_ids": token_ids,
            "operation": "subscribe",
            "custom_feature_enabled": True,
        })
        try:
            await self._ws.send_str(msg)
            logger.debug("Dynamic subscribe: %d tokens", len(token_ids))
        except Exception as e:
            logger.debug("Dynamic subscribe failed: %s", e)

    # ── Message handling ─────────────────────────────────────────────

    def _handle_message(self, raw: str) -> None:
        """Parse and route an incoming WS message."""
        # PONG responses
        if raw == "PONG":
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        self._ws_messages_received += 1

        # Messages can be a list of events or a single event
        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type", "")

            if event_type == "book":
                self._apply_book_snapshot(event)
            elif event_type == "price_change":
                self._apply_price_change(event)
            elif event_type == "best_bid_ask":
                self._apply_best_bid_ask(event)
            elif event_type == "last_trade_price":
                pass  # Informational only, no book update needed
            elif event_type == "tick_size_change":
                pass  # No action needed

    def _apply_book_snapshot(self, event: dict) -> None:
        """Full book snapshot — replace the entire book for this token."""
        asset_id = event.get("asset_id", "")
        if not asset_id or asset_id not in self._tracked:
            return

        meta = self._tracked[asset_id]
        bids_raw = event.get("bids", [])
        asks_raw = event.get("asks", [])

        bid_levels = [
            (float(b.get("price", 0)), float(b.get("size", 0)))
            for b in bids_raw[:10]
        ]
        ask_levels = [
            (float(a.get("price", 0)), float(a.get("size", 0)))
            for a in asks_raw[:10]
        ]

        snap = self._build_snapshot(asset_id, meta, bid_levels, ask_levels)
        with self._lock:
            self._snapshots[asset_id] = snap

    def _apply_price_change(self, event: dict) -> None:
        """Incremental price level update — merge into the existing book."""
        changes = event.get("changes", event.get("price_changes", []))
        if not changes:
            return

        for change in changes:
            asset_id = change.get("asset_id", "")
            if not asset_id or asset_id not in self._tracked:
                continue

            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "").upper()

            if price <= 0 or side not in ("BUY", "SELL"):
                continue

            with self._lock:
                existing = self._snapshots.get(asset_id)
                if not existing:
                    # No book yet; wait for a full snapshot
                    continue

                # Clone levels to mutate
                if side == "BUY":
                    levels = list(existing.bid_levels)
                    levels = self._merge_level(levels, price, size, descending=True)
                    bid_levels = levels
                    ask_levels = list(existing.ask_levels)
                else:
                    levels = list(existing.ask_levels)
                    levels = self._merge_level(levels, price, size, descending=False)
                    ask_levels = levels
                    bid_levels = list(existing.bid_levels)

                meta = self._tracked[asset_id]
                snap = self._build_snapshot(asset_id, meta, bid_levels, ask_levels)
                self._snapshots[asset_id] = snap

    def _apply_best_bid_ask(self, event: dict) -> None:
        """Fast-path update: just the top-of-book changed."""
        asset_id = event.get("asset_id", "")
        if not asset_id or asset_id not in self._tracked:
            return

        best_bid = float(event.get("best_bid", 0))
        best_ask = float(event.get("best_ask", 0))

        with self._lock:
            existing = self._snapshots.get(asset_id)
            if not existing:
                return

            # Update top-of-book without touching deeper levels
            bid_levels = list(existing.bid_levels)
            ask_levels = list(existing.ask_levels)

            if bid_levels:
                bid_levels[0] = (best_bid, bid_levels[0][1])
            else:
                bid_levels = [(best_bid, 0.0)]

            if ask_levels:
                ask_levels[0] = (best_ask, ask_levels[0][1])
            else:
                ask_levels = [(best_ask, 0.0)]

            meta = self._tracked[asset_id]
            snap = self._build_snapshot(asset_id, meta, bid_levels, ask_levels)
            self._snapshots[asset_id] = snap

    @staticmethod
    def _merge_level(
        levels: list[tuple[float, float]],
        price: float,
        size: float,
        descending: bool,
    ) -> list[tuple[float, float]]:
        """Insert or update a price level.  Remove if size == 0."""
        # Remove existing level at this price
        levels = [(p, s) for p, s in levels if abs(p - price) > 1e-9]

        if size > 0:
            levels.append((price, size))

        # Re-sort: bids descending, asks ascending
        levels.sort(key=lambda x: x[0], reverse=descending)

        # Keep top 10 levels
        return levels[:10]

    @staticmethod
    def _build_snapshot(
        token_id: str,
        meta: _TokenMeta,
        bid_levels: list[tuple[float, float]],
        ask_levels: list[tuple[float, float]],
    ) -> BookSnapshot:
        best_bid = bid_levels[0][0] if bid_levels else 0.0
        best_ask = ask_levels[0][0] if ask_levels else 1.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else 0.0
        spread = best_ask - best_bid
        depth = (
            sum(p * s for p, s in bid_levels[:3])
            + sum(p * s for p, s in ask_levels[:3])
        )

        return BookSnapshot(
            timestamp=datetime.now(timezone.utc),
            token_id=token_id,
            condition_id=meta.condition_id,
            outcome=meta.outcome,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            depth_usd=depth,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )

    # ── REST fallback poller ─────────────────────────────────────────

    async def _fallback_poll_loop(self) -> None:
        """Poll REST /book endpoint when WS is disconnected."""
        while self._running:
            if not self._ws_connected:
                await self._poll_all_rest()
            await asyncio.sleep(REST_FALLBACK_INTERVAL_S)

    async def _poll_all_rest(self) -> None:
        """Poll REST endpoint for all tracked tokens (same as CLOBFeed)."""
        if not self._session:
            return

        for token_id, meta in list(self._tracked.items()):
            if self._ws_connected:
                # WS came back mid-poll; stop polling
                return
            try:
                snap = await self._poll_book_rest(token_id, meta)
                if snap:
                    with self._lock:
                        self._snapshots[token_id] = snap
                    self._rest_polls += 1
            except Exception:
                logger.debug("REST poll error for %s", token_id)
            await asyncio.sleep(0.15)

    async def _poll_book_rest(
        self, token_id: str, meta: _TokenMeta
    ) -> BookSnapshot | None:
        """Single REST book fetch (mirrors CLOBFeed._poll_book)."""
        assert self._session is not None
        url = f"{settings.POLYMARKET_HOST}/book"
        try:
            async with self._session.get(
                url,
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        bid_levels = [
            (float(b.get("price", 0)), float(b.get("size", 0)))
            for b in bids[:10]
        ]
        ask_levels = [
            (float(a.get("price", 0)), float(a.get("size", 0)))
            for a in asks[:10]
        ]

        return self._build_snapshot(token_id, meta, bid_levels, ask_levels)

    # ── Convenience (CLOBFeed compat) ────────────────────────────────

    async def run_poll_loop(self) -> None:
        """Compatibility shim: WSFeed uses start() instead, but if called
        from code that expects CLOBFeed.run_poll_loop(), just block."""
        while self._running:
            await asyncio.sleep(1)
