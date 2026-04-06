"""Polymarket CLOB polling for weather temperature markets.

Discovers active temperature contracts, polls order books, and emits
MarketSnapshot dataclasses for the signal engine.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

import aiohttp

from config import settings
from config.locations import LOCATIONS
from core.weather_model import BucketDef

logger = logging.getLogger(__name__)

_TEMP_PATTERN = re.compile(
    r"[Hh]ighest\s+temperature\s+in\s+(.+?)\s+on\s+(\w+\s+\d{1,2})",
    re.IGNORECASE,
)
_BUCKET_PATTERN = re.compile(r"(\d{1,3})\s*[-–]\s*(\d{1,3})\s*°?\s*F", re.IGNORECASE)


@dataclass
class MarketSnapshot:
    timestamp: datetime
    contract_id: str
    location: str
    target_date: date
    bucket_label: str
    bucket_low_f: int
    bucket_high_f: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    depth_usd: float
    volume_24h: float
    time_to_resolution_hrs: float


@dataclass
class ContractMeta:
    condition_id: str
    token_id: str
    location: str
    target_date: date
    bucket_label: str
    bucket_low_f: int
    bucket_high_f: int
    end_date: datetime
    question: str


class PolymarketFeed:
    """Discovers and polls Polymarket weather temperature contracts."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._contracts: dict[str, ContractMeta] = {}
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._callbacks: list[Callable[[MarketSnapshot], Any]] = []
        self._running = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        await self._discover_contracts()

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    def on_snapshot(self, cb: Callable[[MarketSnapshot], Any]) -> None:
        self._callbacks.append(cb)

    def get_snapshot(self, contract_id: str) -> MarketSnapshot | None:
        return self._snapshots.get(contract_id)

    def get_all_snapshots(self) -> dict[str, MarketSnapshot]:
        return dict(self._snapshots)

    def get_bucket_defs(self) -> list[BucketDef]:
        return [
            BucketDef(
                contract_id=c.token_id,
                location=c.location,
                target_date=c.target_date,
                temp_low_f=c.bucket_low_f,
                temp_high_f=c.bucket_high_f,
                label=c.bucket_label,
            )
            for c in self._contracts.values()
        ]

    def get_active_location_dates(self) -> set[tuple[str, date]]:
        return {(c.location, c.target_date) for c in self._contracts.values()}

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def run_poll_loop(self) -> None:
        while self._running:
            await self._poll_all_books()
            await asyncio.sleep(settings.POLYMARKET_POLL_INTERVAL)

    async def run_discovery_loop(self) -> None:
        while self._running:
            await asyncio.sleep(settings.CONTRACT_SCAN_INTERVAL)
            await self._discover_contracts()

    # ------------------------------------------------------------------
    # Contract discovery
    # ------------------------------------------------------------------

    async def _discover_contracts(self) -> None:
        assert self._session is not None
        url = f"{settings.POLYMARKET_HOST}/markets"
        try:
            async with self._session.get(
                url,
                params={"active": "true", "limit": 200},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Polymarket markets endpoint returned %d", resp.status)
                    return
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Polymarket discovery error: %s", exc)
            return

        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        new_count = 0
        for market in markets:
            contracts = self._parse_market(market)
            for c in contracts:
                if c.token_id not in self._contracts:
                    self._contracts[c.token_id] = c
                    new_count += 1

        if new_count:
            logger.info("Discovered %d new weather contracts (total: %d)", new_count, len(self._contracts))

    def _parse_market(self, market: dict[str, Any]) -> list[ContractMeta]:
        question = market.get("question", "") or market.get("title", "")
        match = _TEMP_PATTERN.search(question)
        if not match:
            return []

        city_raw = match.group(1).strip()
        date_raw = match.group(2).strip()

        location = self._match_location(city_raw)
        if not location:
            return []

        target = self._parse_date(date_raw)
        if not target:
            return []

        today = datetime.now(timezone.utc).date()
        lead = (target - today).days
        if lead < 0 or lead > settings.MAX_LEAD_DAYS:
            return []

        end_str = market.get("end_date_iso", "") or market.get("endDate", "")
        try:
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            end_date = datetime.now(timezone.utc)

        outcomes = market.get("tokens", market.get("outcomes", []))
        results: list[ContractMeta] = []
        for outcome in outcomes:
            if isinstance(outcome, dict):
                label = outcome.get("outcome", "") or outcome.get("name", "")
                token_id = outcome.get("token_id", "") or outcome.get("tokenId", "")
            else:
                label = str(outcome)
                token_id = str(outcome)

            bucket_match = _BUCKET_PATTERN.search(label)
            if not bucket_match:
                continue

            low_f = int(bucket_match.group(1))
            high_f = int(bucket_match.group(2))
            condition_id = market.get("condition_id", "") or market.get("conditionId", "")

            results.append(ContractMeta(
                condition_id=condition_id,
                token_id=token_id,
                location=location,
                target_date=target,
                bucket_label=label,
                bucket_low_f=low_f,
                bucket_high_f=high_f,
                end_date=end_date,
                question=question,
            ))
        return results

    @staticmethod
    def _match_location(city_raw: str) -> str | None:
        city_lower = city_raw.lower()
        for key, cfg in LOCATIONS.items():
            if cfg.name.lower() in city_lower or key.lower() in city_lower:
                return key
        alias_map = {
            "new york": "NYC", "nyc": "NYC", "manhattan": "NYC",
            "chicago": "Chicago", "chi": "Chicago",
            "seattle": "Seattle", "sea": "Seattle",
            "atlanta": "Atlanta", "atl": "Atlanta",
            "dallas": "Dallas", "dal": "Dallas", "dfw": "Dallas",
        }
        return alias_map.get(city_lower)

    @staticmethod
    def _parse_date(date_raw: str) -> date | None:
        import calendar
        parts = date_raw.split()
        if len(parts) < 2:
            return None
        month_str = parts[0][:3].title()
        try:
            month_num = list(calendar.month_abbr).index(month_str)
        except ValueError:
            return None
        try:
            day = int(parts[1])
        except ValueError:
            return None
        year = datetime.now().year
        try:
            return date(year, month_num, day)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Order book polling
    # ------------------------------------------------------------------

    async def _poll_all_books(self) -> None:
        now = datetime.now(timezone.utc)
        for token_id, meta in list(self._contracts.items()):
            ttl_hrs = (meta.end_date - now).total_seconds() / 3600.0
            if ttl_hrs < 0:
                self._contracts.pop(token_id, None)
                continue
            try:
                snap = await self._poll_book(token_id, meta, ttl_hrs)
                if snap:
                    self._snapshots[token_id] = snap
                    for cb in self._callbacks:
                        try:
                            cb(snap)
                        except Exception:
                            logger.exception("Snapshot callback error")
            except Exception:
                logger.exception("Error polling book for %s", token_id)
            await asyncio.sleep(0.2)

    async def _poll_book(
        self, token_id: str, meta: ContractMeta, ttl_hrs: float
    ) -> MarketSnapshot | None:
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

        best_bid = float(bids[0].get("price", 0)) if bids else 0.0
        best_ask = float(asks[0].get("price", 1)) if asks else 1.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else 0.0
        spread = best_ask - best_bid

        depth = 0.0
        for level in (bids[:3] + asks[:3]):
            depth += float(level.get("size", 0)) * float(level.get("price", 0))

        volume = float(data.get("volume_24h", data.get("volume", 0)))

        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            contract_id=token_id,
            location=meta.location,
            target_date=meta.target_date,
            bucket_label=meta.bucket_label,
            bucket_low_f=meta.bucket_low_f,
            bucket_high_f=meta.bucket_high_f,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            depth_usd=depth,
            volume_24h=volume,
            time_to_resolution_hrs=ttl_hrs,
        )
