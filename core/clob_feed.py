"""CLOB API feed for real-time order book polling.

Polls order books for tracked markets at high frequency. Provides
bid/ask/mid prices, depth, and spread for the signal engine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class BookSnapshot:
    """Real-time order book snapshot for a single token."""

    timestamp: datetime
    token_id: str
    condition_id: str
    outcome: str
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    depth_usd: float
    bid_levels: list[tuple[float, float]]  # [(price, size)]
    ask_levels: list[tuple[float, float]]


class CLOBFeed:
    """Polls Polymarket CLOB for real-time order book data."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._tracked_tokens: dict[str, dict[str, str]] = {}  # token_id -> {condition_id, outcome}
        self._snapshots: dict[str, BookSnapshot] = {}  # token_id -> latest snapshot
        self._running = False

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    def track_token(self, token_id: str, condition_id: str, outcome: str) -> None:
        """Add a token to the tracking set."""
        self._tracked_tokens[token_id] = {"condition_id": condition_id, "outcome": outcome}

    def untrack_token(self, token_id: str) -> None:
        self._tracked_tokens.pop(token_id, None)
        self._snapshots.pop(token_id, None)

    def clear_tracked(self) -> None:
        self._tracked_tokens.clear()
        self._snapshots.clear()

    def get_snapshot(self, token_id: str) -> BookSnapshot | None:
        return self._snapshots.get(token_id)

    def get_all_snapshots(self) -> dict[str, BookSnapshot]:
        return dict(self._snapshots)

    async def run_poll_loop(self) -> None:
        """Poll order books for all tracked tokens."""
        while self._running:
            await self._poll_all()
            await asyncio.sleep(settings.CLOB_POLL_INTERVAL)

    async def _poll_all(self) -> None:
        for token_id, meta in list(self._tracked_tokens.items()):
            try:
                snap = await self._poll_book(token_id, meta)
                if snap:
                    self._snapshots[token_id] = snap
            except Exception:
                logger.debug("Error polling book for %s", token_id)
            await asyncio.sleep(0.15)

    async def _poll_book(
        self, token_id: str, meta: dict[str, str]
    ) -> BookSnapshot | None:
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

        bid_levels = [(float(b.get("price", 0)), float(b.get("size", 0))) for b in bids[:5]]
        ask_levels = [(float(a.get("price", 0)), float(a.get("size", 0))) for a in asks[:5]]

        best_bid = bid_levels[0][0] if bid_levels else 0.0
        best_ask = ask_levels[0][0] if ask_levels else 1.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else 0.0
        spread = best_ask - best_bid

        depth = sum(p * s for p, s in bid_levels[:3]) + sum(p * s for p, s in ask_levels[:3])

        return BookSnapshot(
            timestamp=datetime.now(timezone.utc),
            token_id=token_id,
            condition_id=meta["condition_id"],
            outcome=meta["outcome"],
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            depth_usd=depth,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )
