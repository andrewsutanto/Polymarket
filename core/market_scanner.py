"""Market scanner orchestrator.

Coordinates Gamma discovery + CLOB polling + strategy execution.
Ranks opportunities and manages the tracked market set.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from config import settings
from core.gamma_feed import GammaFeed, GammaMarket
from core.clob_feed import CLOBFeed, BookSnapshot
from core.fair_value import structural_fair_value, complement_fair_value
from core.relationship_graph import find_relationships, MarketRelationship
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    """A ranked trading opportunity."""

    market_id: str
    token_id: str
    market_slug: str
    question: str
    category: str
    outcome: str
    strategy_name: str
    direction: str
    edge: float
    strength: float
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    ttl_hours: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketScanner:
    """Orchestrate universe discovery, tracking, and signal generation."""

    def __init__(
        self,
        gamma: GammaFeed,
        clob: CLOBFeed,
        strategies: list[BaseStrategy],
    ) -> None:
        self._gamma = gamma
        self._clob = clob
        self._strategies = strategies
        self._relationships: list[MarketRelationship] = []
        self._opportunities: list[Opportunity] = []
        self._signal_callbacks: list[Callable[[Signal], Any]] = []
        self._running = False

    @property
    def opportunities(self) -> list[Opportunity]:
        return list(self._opportunities)

    @property
    def relationships(self) -> list[MarketRelationship]:
        return list(self._relationships)

    @property
    def tracked_count(self) -> int:
        return self._clob.snapshot_count

    def on_signal(self, cb: Callable[[Signal], Any]) -> None:
        self._signal_callbacks.append(cb)

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def full_scan(self) -> list[Opportunity]:
        """Run a full universe scan: discover markets, find relationships,
        rank opportunities, and promote top N to CLOB tracking.

        Returns:
            Ranked list of opportunities.
        """
        # Refresh universe
        await self._gamma.scan_universe()
        markets = self._gamma.markets

        if not markets:
            logger.warning("No markets found in universe scan")
            return []

        # Find cross-market relationships
        self._relationships = find_relationships(markets)

        # Run strategies on all markets (using Gamma prices)
        opportunities: list[Opportunity] = []
        for condition_id, market in markets.items():
            opps = self._evaluate_market(market)
            opportunities.extend(opps)

        # Sort by edge * strength (best opportunities first)
        opportunities.sort(key=lambda o: o.edge * o.strength, reverse=True)
        self._opportunities = opportunities[:200]

        # Promote top N to CLOB tracking
        self._clob.clear_tracked()
        promoted = set()
        for opp in opportunities[:settings.CLOB_TOP_N_TRACKED]:
            if opp.token_id and opp.token_id not in promoted:
                self._clob.track_token(opp.token_id, opp.market_id, opp.outcome)
                promoted.add(opp.token_id)

        logger.info(
            "Full scan: %d opportunities found, %d promoted to CLOB tracking",
            len(opportunities), len(promoted),
        )
        return self._opportunities

    async def run_signal_scan(self) -> list[Signal]:
        """Run strategies on tracked markets using real-time CLOB data.

        Returns:
            List of generated signals.
        """
        signals: list[Signal] = []
        snapshots = self._clob.get_all_snapshots()

        if not snapshots:
            return signals

        for token_id, snap in snapshots.items():
            gamma_market = self._gamma.get_market(snap.condition_id)
            if not gamma_market:
                continue

            market_data = self._build_market_data(gamma_market, snap)

            for strategy in self._strategies:
                try:
                    signal = strategy.generate_signal(market_data)
                    if signal is not None:
                        signals.append(signal)
                        for cb in self._signal_callbacks:
                            try:
                                cb(signal)
                            except Exception:
                                logger.exception("Signal callback error")
                except Exception:
                    logger.debug("Strategy %s error on %s", strategy.name, token_id)

        return signals

    async def run_scan_loop(self) -> None:
        """Continuous signal scanning loop."""
        while self._running:
            try:
                await self.run_signal_scan()
            except Exception:
                logger.exception("Error in signal scan loop")
            await asyncio.sleep(settings.SCAN_INTERVAL)

    async def run_full_scan_loop(self) -> None:
        """Periodic full universe scan."""
        while self._running:
            try:
                await self.full_scan()
            except Exception:
                logger.exception("Error in full scan loop")
            await asyncio.sleep(settings.FULL_SCAN_INTERVAL)

    def _evaluate_market(self, market: GammaMarket) -> list[Opportunity]:
        """Run all strategies on a single market using Gamma data."""
        opps: list[Opportunity] = []

        if len(market.outcome_prices) < 2:
            return opps

        yes_price = market.outcome_prices[0]
        no_price = market.outcome_prices[1] if len(market.outcome_prices) > 1 else 1.0 - yes_price

        ttl_hours = 0.0
        if market.end_date:
            delta = (market.end_date - datetime.now(timezone.utc)).total_seconds()
            ttl_hours = max(delta / 3600.0, 0)

        # Build basic market data for strategy evaluation
        for i, token_info in enumerate(market.tokens):
            price = market.outcome_prices[i] if i < len(market.outcome_prices) else 0.5
            complement_price = 1.0 - price  # Simplified for binary markets

            market_data: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc),
                "market_id": market.condition_id,
                "token_id": token_info.get("token_id", ""),
                "market_slug": market.slug,
                "category": market.category,
                "question": market.question,
                "outcome": token_info.get("outcome", f"Outcome_{i}"),
                "yes_price": yes_price,
                "no_price": no_price,
                "mid_price": price,
                "complement_price": complement_price,
                "spread": abs(yes_price + no_price - 1.0),
                "book_depth": 0.0,
                "volume_24h": market.volume_24h,
                "liquidity": market.liquidity,
                "time_to_resolution_hrs": ttl_hours,
                "has_position": False,
                "outcomes": market.outcomes,
                "outcome_prices": market.outcome_prices,
                "tags": market.tags,
                "relationships": [r for r in self._relationships
                                  if r.market_a_id == market.condition_id or r.market_b_id == market.condition_id],
            }

            for strategy in self._strategies:
                try:
                    signal = strategy.generate_signal(market_data)
                    if signal is not None and signal.edge >= settings.MIN_EDGE:
                        opps.append(Opportunity(
                            market_id=market.condition_id,
                            token_id=token_info.get("token_id", ""),
                            market_slug=market.slug,
                            question=market.question,
                            category=market.category,
                            outcome=token_info.get("outcome", ""),
                            strategy_name=signal.strategy_name,
                            direction=signal.direction,
                            edge=signal.edge,
                            strength=signal.strength,
                            yes_price=yes_price,
                            no_price=no_price,
                            volume_24h=market.volume_24h,
                            liquidity=market.liquidity,
                            ttl_hours=ttl_hours,
                        ))
                except Exception:
                    pass

        return opps

    def _build_market_data(
        self, market: GammaMarket, snap: BookSnapshot
    ) -> dict[str, Any]:
        """Build market_data dict from Gamma + CLOB data."""
        ttl_hours = 0.0
        if market.end_date:
            delta = (market.end_date - datetime.now(timezone.utc)).total_seconds()
            ttl_hours = max(delta / 3600.0, 0)

        yes_price = market.outcome_prices[0] if market.outcome_prices else 0.5
        no_price = market.outcome_prices[1] if len(market.outcome_prices) > 1 else 1.0 - yes_price

        return {
            "timestamp": snap.timestamp,
            "market_id": market.condition_id,
            "token_id": snap.token_id,
            "market_slug": market.slug,
            "category": market.category,
            "question": market.question,
            "outcome": snap.outcome,
            "yes_price": yes_price,
            "no_price": no_price,
            "mid_price": snap.mid_price,
            "best_bid": snap.best_bid,
            "best_ask": snap.best_ask,
            "complement_price": 1.0 - snap.mid_price,
            "spread": snap.spread,
            "book_depth": snap.depth_usd,
            "volume_24h": market.volume_24h,
            "liquidity": market.liquidity,
            "time_to_resolution_hrs": ttl_hours,
            "has_position": False,
            "outcomes": market.outcomes,
            "outcome_prices": market.outcome_prices,
            "tags": market.tags,
            "bid_levels": snap.bid_levels,
            "ask_levels": snap.ask_levels,
            "relationships": [r for r in self._relationships
                              if r.market_a_id == market.condition_id or r.market_b_id == market.condition_id],
        }
