"""Weather contract universe discovery and filtering.

Scans Polymarket API for active weather-related contracts and returns
a structured list of tradeable opportunities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

WEATHER_KEYWORDS = [
    "temperature", "highest temp", "lowest temp", "high temp",
    "precipitation", "rain", "snow", "wind", "weather",
]

TEMP_PATTERN = re.compile(
    r"(highest|lowest|high|low)\s+temp\w*\s+in\s+(.+?)\s+on\s+",
    re.IGNORECASE,
)


@dataclass
class WeatherContract:
    """A tradeable weather contract from Polymarket."""

    market_id: str
    question: str
    city: str | None
    contract_type: str  # "temperature", "precipitation", "wind", "other"
    expiry: datetime | None
    current_odds: float
    volume_24h: float
    spread: float
    liquidity: float
    weather_variable: str
    url: str


class UniverseScanner:
    """Scan Polymarket for active weather contracts."""

    def __init__(
        self,
        min_liquidity: float = 10.0,
        min_volume: float = 5.0,
        max_spread: float = 0.15,
        contract_types: list[str] | None = None,
        cities: list[str] | None = None,
        max_expiry_days: int = 7,
    ) -> None:
        self._min_liq = min_liquidity
        self._min_vol = min_volume
        self._max_spread = max_spread
        self._types = contract_types or ["temperature", "precipitation", "wind"]
        self._cities = [c.lower() for c in cities] if cities else None
        self._max_expiry = max_expiry_days

    def scan(self) -> list[WeatherContract]:
        """Scan Polymarket for active weather contracts.

        Returns:
            Filtered list of WeatherContract objects.
        """
        url = f"{settings.POLYMARKET_HOST}/markets"
        try:
            resp = requests.get(url, params={"active": "true", "limit": 500}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("Universe scan failed: %s", exc)
            return []

        markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
        contracts: list[WeatherContract] = []

        for m in markets:
            question = (m.get("question", "") or m.get("title", "")).lower()
            if not any(kw in question for kw in WEATHER_KEYWORDS):
                continue

            contract = self._parse_contract(m)
            if contract and self._passes_filters(contract):
                contracts.append(contract)

        logger.info("Universe scan: %d weather contracts found", len(contracts))
        return contracts

    def _parse_contract(self, market: dict[str, Any]) -> WeatherContract | None:
        question = market.get("question", "") or market.get("title", "")
        market_id = market.get("condition_id", "") or market.get("id", "")

        # Detect type
        q_lower = question.lower()
        if any(kw in q_lower for kw in ("temperature", "temp", "degrees")):
            ctype = "temperature"
        elif any(kw in q_lower for kw in ("rain", "precipitation", "inches")):
            ctype = "precipitation"
        elif "wind" in q_lower:
            ctype = "wind"
        else:
            ctype = "other"

        # Extract city
        match = TEMP_PATTERN.search(question)
        city = match.group(2).strip() if match else None

        # Expiry
        end_str = market.get("end_date_iso", "") or market.get("endDate", "")
        try:
            expiry = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else None
        except ValueError:
            expiry = None

        # Market data
        tokens = market.get("tokens", [])
        odds = 0.5
        if tokens and isinstance(tokens[0], dict):
            odds = float(tokens[0].get("price", 0.5))

        return WeatherContract(
            market_id=market_id,
            question=question,
            city=city,
            contract_type=ctype,
            expiry=expiry,
            current_odds=odds,
            volume_24h=float(market.get("volume", market.get("volume24h", 0))),
            spread=float(market.get("spread", 0.05)),
            liquidity=float(market.get("liquidity", 0)),
            weather_variable=ctype,
            url=f"https://polymarket.com/event/{market_id}",
        )

    def _passes_filters(self, c: WeatherContract) -> bool:
        if c.contract_type not in self._types:
            return False
        if c.liquidity < self._min_liq:
            return False
        if c.volume_24h < self._min_vol:
            return False
        if c.spread > self._max_spread:
            return False
        if self._cities and c.city and c.city.lower() not in self._cities:
            return False
        if c.expiry:
            days_to_expiry = (c.expiry - datetime.now(timezone.utc)).days
            if days_to_expiry > self._max_expiry or days_to_expiry < 0:
                return False
        return True
