"""Falcon API integration for smart money, sentiment, and cross-market data.

Enriches signals with institutional flow, social sentiment, and Kalshi
price comparisons. Gracefully degrades if Falcon is unavailable.
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
class FalconIntel:
    timestamp: datetime
    contract_slug: str
    smart_money_bias: float
    whale_activity: bool
    top_trader_agreement: float
    sentiment_score: float
    cross_market_gap: float
    kalshi_confirms_direction: bool


def neutral_intel(contract_slug: str) -> FalconIntel:
    """Return a neutral FalconIntel when the API is unavailable."""
    return FalconIntel(
        timestamp=datetime.now(timezone.utc),
        contract_slug=contract_slug,
        smart_money_bias=0.0,
        whale_activity=False,
        top_trader_agreement=0.0,
        sentiment_score=0.0,
        cross_market_gap=0.0,
        kalshi_confirms_direction=False,
    )


class FalconFeed:
    """Polls Falcon API for smart-money, sentiment, and cross-market data."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._cache: dict[str, FalconIntel] = {}
        self._top_traders: list[dict[str, Any]] = []
        self._enabled = settings.FALCON_ENABLED and bool(settings.FALCON_API_TOKEN)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        if not self._enabled:
            logger.warning("Falcon API disabled (no token or FALCON_ENABLED=false)")
            return
        headers = {
            "Authorization": f"Bearer {settings.FALCON_API_TOKEN}",
            "Content-Type": "application/json",
        }
        self._session = aiohttp.ClientSession(headers=headers)
        self._running = True
        await self._fetch_top_traders()

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    def get_intel(self, contract_slug: str) -> FalconIntel:
        return self._cache.get(contract_slug, neutral_intel(contract_slug))

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def run_smart_money_loop(self) -> None:
        while self._running and self._enabled:
            await self._poll_smart_money()
            await asyncio.sleep(settings.FALCON_SMART_MONEY_INTERVAL)

    async def run_sentiment_loop(self) -> None:
        while self._running and self._enabled:
            await self._poll_sentiment()
            await asyncio.sleep(settings.FALCON_SENTIMENT_INTERVAL)

    async def run_cross_market_loop(self) -> None:
        while self._running and self._enabled:
            await self._poll_cross_market()
            await asyncio.sleep(settings.FALCON_CROSS_MARKET_INTERVAL)

    # ------------------------------------------------------------------
    # Smart money
    # ------------------------------------------------------------------

    async def _fetch_top_traders(self) -> None:
        data = await self._post("traders/stats", {
            "limit": settings.FALCON_TOP_TRADERS_COUNT,
            "sort_by": "f_score",
        })
        if data and isinstance(data, list):
            self._top_traders = data
            logger.info("Loaded %d top Falcon traders", len(self._top_traders))
        elif data and isinstance(data, dict):
            self._top_traders = data.get("traders", [])

    async def _poll_smart_money(self) -> None:
        if not self._top_traders:
            return
        for trader in self._top_traders[:10]:
            wallet = trader.get("address", trader.get("wallet", ""))
            if not wallet:
                continue
            data = await self._post("traders/positions", {"address": wallet})
            if not data:
                continue
            positions = data if isinstance(data, list) else data.get("positions", [])
            for pos in positions:
                slug = pos.get("slug", pos.get("market_slug", ""))
                if not slug:
                    continue
                existing = self._cache.get(slug, neutral_intel(slug))
                side = pos.get("side", "")
                size = float(pos.get("size", 0))
                if side.upper() == "YES" and size > 0:
                    existing.smart_money_bias = min(existing.smart_money_bias + 0.1, 1.0)
                elif side.upper() == "NO" and size > 0:
                    existing.smart_money_bias = max(existing.smart_money_bias - 0.1, -1.0)
                existing.whale_activity = True
                existing.top_trader_agreement = max(existing.top_trader_agreement, 0.5)
                existing.timestamp = datetime.now(timezone.utc)
                self._cache[slug] = existing
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    async def _poll_sentiment(self) -> None:
        for slug in list(self._cache.keys()):
            data = await self._post("signals/sentiment", {"slug": slug})
            if not data:
                continue
            existing = self._cache.get(slug, neutral_intel(slug))
            existing.sentiment_score = float(data.get("sentiment_score", 0))
            existing.timestamp = datetime.now(timezone.utc)
            self._cache[slug] = existing
            await asyncio.sleep(0.3)

    # ------------------------------------------------------------------
    # Cross-market (Polymarket vs Kalshi)
    # ------------------------------------------------------------------

    async def _poll_cross_market(self) -> None:
        for slug in list(self._cache.keys()):
            data = await self._post("cross/compare", {"slug": slug})
            if not data:
                continue
            existing = self._cache.get(slug, neutral_intel(slug))
            price_gap = float(data.get("price_gap", 0))
            existing.cross_market_gap = price_gap
            existing.kalshi_confirms_direction = abs(price_gap) > 0.05
            existing.timestamp = datetime.now(timezone.utc)
            self._cache[slug] = existing
            await asyncio.sleep(0.3)

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any] | list | None:
        if not self._session:
            return None
        url = f"{settings.FALCON_API_BASE}/{endpoint}"
        try:
            async with self._session.post(
                url, json=body, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.debug("Falcon %s returned %d", endpoint, resp.status)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("Falcon %s error: %s", endpoint, exc)
        return None
