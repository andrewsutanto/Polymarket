"""Gamma API feed for Polymarket universe discovery.

Polls gamma-api.polymarket.com for all active markets with prices,
volume, liquidity, tags, and expiry. No API key required.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class GammaMarket:
    """A market discovered via the Gamma API."""

    condition_id: str
    question: str
    slug: str
    category: str
    end_date: datetime | None
    active: bool
    closed: bool
    liquidity: float
    volume_24h: float
    volume_total: float
    outcomes: list[str]
    outcome_prices: list[float]
    tokens: list[dict[str, str]]
    tags: list[str]
    description: str
    image: str
    icon: str
    last_updated: datetime


class GammaFeed:
    """Polls Gamma API for Polymarket universe discovery."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._markets: dict[str, GammaMarket] = {}
        self._running = False
        self._last_scan: datetime | None = None

    @property
    def markets(self) -> dict[str, GammaMarket]:
        return dict(self._markets)

    @property
    def market_count(self) -> int:
        return len(self._markets)

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        await self.scan_universe()

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    async def run_poll_loop(self) -> None:
        """Poll Gamma API on schedule."""
        while self._running:
            await asyncio.sleep(settings.GAMMA_POLL_INTERVAL)
            await self.scan_universe()

    async def scan_universe(self) -> int:
        """Fetch all active markets from Gamma API.

        Returns:
            Number of active markets found.
        """
        assert self._session is not None
        all_markets: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                }
                async with self._session.get(
                    f"{settings.GAMMA_API_BASE}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("Gamma API returned %d", resp.status)
                        break
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Gamma API error: %s", exc)
                break

            if not data:
                break

            all_markets.extend(data)
            offset += limit

            if len(data) < limit:
                break

            # Safety: don't fetch more than 5000 markets
            if offset >= 5000:
                break

            await asyncio.sleep(0.3)

        # Parse and filter
        new_count = 0
        for raw in all_markets:
            market = self._parse_market(raw)
            if market and self._passes_filters(market):
                if market.condition_id not in self._markets:
                    new_count += 1
                self._markets[market.condition_id] = market

        self._last_scan = datetime.now(timezone.utc)
        logger.info(
            "Gamma scan: %d total active markets (%d new)",
            len(self._markets), new_count,
        )
        return len(self._markets)

    def get_market(self, condition_id: str) -> GammaMarket | None:
        return self._markets.get(condition_id)

    def get_markets_by_category(self, category: str) -> list[GammaMarket]:
        return [m for m in self._markets.values() if m.category == category]

    def get_top_by_volume(self, n: int = 50) -> list[GammaMarket]:
        sorted_m = sorted(self._markets.values(), key=lambda m: m.volume_24h, reverse=True)
        return sorted_m[:n]

    def get_top_by_liquidity(self, n: int = 50) -> list[GammaMarket]:
        sorted_m = sorted(self._markets.values(), key=lambda m: m.liquidity, reverse=True)
        return sorted_m[:n]

    def _parse_market(self, raw: dict[str, Any]) -> GammaMarket | None:
        try:
            condition_id = raw.get("conditionId") or raw.get("condition_id", "")
            if not condition_id:
                return None

            end_str = raw.get("endDate", "") or raw.get("end_date_iso", "")
            end_date = None
            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Parse outcome prices
            price_str = raw.get("outcomePrices", "[]")
            if isinstance(price_str, str):
                import json
                try:
                    prices = [float(p) for p in json.loads(price_str)]
                except (json.JSONDecodeError, ValueError):
                    prices = []
            elif isinstance(price_str, list):
                prices = [float(p) for p in price_str]
            else:
                prices = []

            outcomes = raw.get("outcomes", [])
            if isinstance(outcomes, str):
                import json
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = []

            # Parse tokens
            tokens_raw = raw.get("clobTokenIds", "[]")
            if isinstance(tokens_raw, str):
                import json
                try:
                    token_ids = json.loads(tokens_raw)
                except json.JSONDecodeError:
                    token_ids = []
            elif isinstance(tokens_raw, list):
                token_ids = tokens_raw
            else:
                token_ids = []

            tokens = []
            for i, tid in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"Outcome_{i}"
                price = prices[i] if i < len(prices) else 0.0
                tokens.append({"token_id": str(tid), "outcome": outcome, "price": str(price)})

            tags_raw = raw.get("tags", [])
            if isinstance(tags_raw, str):
                import json
                try:
                    tags = json.loads(tags_raw)
                except json.JSONDecodeError:
                    tags = []
            else:
                tags = tags_raw or []

            from core.market_classifier import classify_market
            category = classify_market(
                raw.get("question", ""),
                raw.get("description", ""),
                tags,
            )

            return GammaMarket(
                condition_id=condition_id,
                question=raw.get("question", ""),
                slug=raw.get("slug", condition_id[:20]),
                category=category,
                end_date=end_date,
                active=bool(raw.get("active", True)),
                closed=bool(raw.get("closed", False)),
                liquidity=float(raw.get("liquidity", 0) or 0),
                volume_24h=float(raw.get("volume24hr", raw.get("volume_24h", 0)) or 0),
                volume_total=float(raw.get("volume", 0) or 0),
                outcomes=outcomes if isinstance(outcomes, list) else [],
                outcome_prices=prices,
                tokens=tokens,
                tags=[str(t) for t in tags],
                description=raw.get("description", "")[:500],
                image=raw.get("image", ""),
                icon=raw.get("icon", ""),
                last_updated=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.debug("Failed to parse market: %s", exc)
            return None

    def _passes_filters(self, m: GammaMarket) -> bool:
        if not m.active or m.closed:
            return False
        if m.liquidity < settings.MIN_MARKET_LIQUIDITY:
            return False
        if m.volume_24h < settings.MIN_MARKET_VOLUME_24H:
            return False
        if not m.outcome_prices:
            return False
        return True
