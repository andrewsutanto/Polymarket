#!/usr/bin/env python3
"""Phase 1 Backtest: Markov Chain + Bias Calibration + Alpha Combination.

Fetches real historical price data from Polymarket APIs,
builds Markov transition matrices, runs Monte Carlo simulations,
applies longshot bias calibration, and backtests multiple strategies
with the Alpha Combination Engine.

Outputs detailed performance metrics and comparison tables.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp

from core.markov_model import MarkovModel, MarkovEstimate
from core.bias_calibrator import BiasCalibrator, CATEGORY_EDGE_MULTIPLIERS
from core.alpha_combiner import AlphaCombiner

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ─── Data Fetching ──────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class MarketData:
    """Resolved market with price history for backtesting."""
    condition_id: str
    question: str
    slug: str
    category: str
    outcomes: list[str]
    outcome_prices: list[float]  # Final resolution prices
    tokens: list[dict]
    volume: float
    end_date: str
    price_history: list[dict] = field(default_factory=list)  # [{"t": ts, "p": price}]
    resolved_yes: bool = False


async def fetch_resolved_markets(
    session: aiohttp.ClientSession, n: int = 100
) -> list[dict]:
    """Fetch resolved markets from Gamma API, sorted by volume."""
    all_markets = []
    offset = 0
    limit = 100

    while len(all_markets) < n:
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        try:
            async with session.get(
                f"{GAMMA_API}/markets", params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
        except Exception as e:
            logger.warning(f"Gamma API error: {e}")
            break

        if not data:
            break
        all_markets.extend(data)
        offset += limit
        if len(data) < limit:
            break
        await asyncio.sleep(0.3)

    return all_markets[:n]


async def fetch_price_history(
    session: aiohttp.ClientSession, token_id: str
) -> list[dict]:
    """Fetch price history for a token from CLOB API."""
    try:
        params = {"market": token_id, "interval": "max", "fidelity": "60"}
        async with session.get(
            f"{CLOB_API}/prices-history", params=params,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("history", [])
    except Exception:
        return []


async def collect_backtest_data(max_markets: int = 200) -> list[MarketData]:
    """Collect resolved markets with price history for backtesting."""
    logger.info("=" * 70)
    logger.info("PHASE 1 BACKTEST — DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Fetching up to {max_markets} resolved markets from Polymarket...")

    async with aiohttp.ClientSession() as session:
        raw_markets = await fetch_resolved_markets(session, max_markets)
        logger.info(f"Found {len(raw_markets)} resolved markets")

        markets = []
        fetched = 0
        skipped_no_tokens = 0
        skipped_no_history = 0

        for raw in raw_markets:
            condition_id = raw.get("conditionId", "")
            question = raw.get("question", "")[:80]
            volume = float(raw.get("volume", 0) or 0)

            # Parse token IDs
            tokens_raw = raw.get("clobTokenIds", "[]")
            if isinstance(tokens_raw, str):
                try:
                    token_ids = json.loads(tokens_raw)
                except json.JSONDecodeError:
                    token_ids = []
            elif isinstance(tokens_raw, list):
                token_ids = tokens_raw
            else:
                token_ids = []

            if not token_ids:
                skipped_no_tokens += 1
                continue

            # Parse outcomes and prices
            outcomes = raw.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = ["Yes", "No"]

            price_str = raw.get("outcomePrices", "[]")
            if isinstance(price_str, str):
                try:
                    prices = [float(p) for p in json.loads(price_str)]
                except (json.JSONDecodeError, ValueError):
                    prices = []
            elif isinstance(price_str, list):
                prices = [float(p) for p in price_str]
            else:
                prices = []

            # Determine resolution
            resolved_yes = prices[0] > 0.5 if prices else False

            # Classify category
            text = f"{question} {raw.get('description', '')}".lower()
            tags = raw.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = []

            category = _classify(text, [str(t) for t in (tags or [])])

            # Fetch price history for first token
            token_id = str(token_ids[0])
            history = await fetch_price_history(session, token_id)
            await asyncio.sleep(0.15)

            if len(history) < 10:
                skipped_no_history += 1
                continue

            tokens = [{"token_id": str(tid)} for tid in token_ids]
            markets.append(MarketData(
                condition_id=condition_id,
                question=question,
                slug=raw.get("slug", ""),
                category=category,
                outcomes=outcomes,
                outcome_prices=prices,
                tokens=tokens,
                volume=volume,
                end_date=raw.get("endDate", ""),
                price_history=history,
                resolved_yes=resolved_yes,
            ))

            fetched += 1
            if fetched % 10 == 0:
                logger.info(f"  Fetched {fetched} markets with history...")

            if fetched >= 80:  # Cap for reasonable runtime
                break

        logger.info(f"\nData collection complete:")
        logger.info(f"  Markets with history: {len(markets)}")
        logger.info(f"  Skipped (no tokens): {skipped_no_tokens}")
        logger.info(f"  Skipped (no history): {skipped_no_history}")

        return markets


def _classify(text: str, tags: list[str]) -> str:
    """Simple category classifier."""
    combined = f"{text} {' '.join(tags)}".lower()
    categories = {
        "sports": ["nba", "nfl", "mlb", "nhl", "ufc", "tennis", "game", "match", "win", "vs"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "xrp"],
        "politics": ["president", "election", "trump", "biden", "congress", "vote", "pm ", "minister"],
        "entertainment": ["oscar", "grammy", "movie", "album", "netflix", "tiktok"],
        "macro": ["fed", "interest rate", "inflation", "gdp", "tariff", "s&p"],
        "science": ["ai ", "openai", "space", "nasa", "climate", "fda"],
    }
    for cat, keywords in categories.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "other"


# ─── Backtest Engine ───────────────────────────────────────────────

@dataclass
class Trade:
    """A simulated trade."""
    market_id: str
    question: str
    category: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float  # Resolution price (0 or 1)
    size_usd: float
    edge_at_entry: float
    strategy: str
    pnl: float = 0.0
    return_pct: float = 0.0

    def compute_pnl(self):
        if self.direction == "BUY":
            # Bought YES at entry_price, resolves to exit_price
            shares = self.size_usd / self.entry_price if self.entry_price > 0 else 0
            self.pnl = shares * self.exit_price - self.size_usd
        else:
            # Bought NO at (1 - entry_price), NO pays (1 - exit_price)
            no_price = 1.0 - self.entry_price
            shares = self.size_usd / no_price if no_price > 0 else 0
            self.pnl = shares * (1.0 - self.exit_price) - self.size_usd
        self.return_pct = self.pnl / self.size_usd if self.size_usd > 0 else 0


def kelly_size(edge: float, price: float, bankroll: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly position sizing."""
    if price <= 0.01 or price >= 0.99 or edge <= 0:
        return 0.0
    odds = (1.0 / price) - 1.0
    if odds <= 0:
        return 0.0
    p = price + edge
    p = min(p, 0.99)
    q = 1.0 - p
    kelly = (p * odds - q) / odds
    kelly *= fraction
    if kelly <= 0:
        return 0.0
    size = kelly * bankroll
    return max(0.50, min(size, 5.0))  # $0.50 min, $5.00 max


def run_backtest(markets: list[MarketData]) -> dict:
    """Run the Phase 1 backtest across all markets.

    Tests 4 approaches:
    1. Naive (buy when price < 0.50, no model)
    2. Markov-only (MC probability estimate)
    3. Markov + Bias Calibration
    4. Full system (Markov + Bias + Alpha Combination + Category)
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 BACKTEST — SIMULATION")
    logger.info("=" * 70)

    markov = MarkovModel(n_states=10, n_simulations=5000)
    calibrator = BiasCalibrator()
    combiner = AlphaCombiner(min_trades_for_ic=10)

    # Register strategies
    for name in ["markov_mc", "markov_absorb", "implied_prob", "no_bias", "maker_edge"]:
        combiner.register_strategy(name)

    results = {
        "naive": {"trades": [], "bankroll": 50.0, "peak": 50.0},
        "markov_only": {"trades": [], "bankroll": 50.0, "peak": 50.0},
        "markov_calibrated": {"trades": [], "bankroll": 50.0, "peak": 50.0},
        "full_system": {"trades": [], "bankroll": 50.0, "peak": 50.0},
    }

    total_markets = len(markets)
    logger.info(f"Running backtest on {total_markets} resolved markets...\n")

    for idx, market in enumerate(markets):
        prices = [h["p"] for h in market.price_history]
        if len(prices) < 10:
            continue

        current_price = prices[-1]
        resolution = 1.0 if market.resolved_yes else 0.0

        # Skip markets at extreme prices (already resolved in practice)
        if current_price > 0.95 or current_price < 0.05:
            # Use price from 70% through history as "entry point"
            entry_idx = int(len(prices) * 0.7)
            if entry_idx < 10:
                continue
            current_price = prices[entry_idx]
            history_for_model = prices[:entry_idx]
        else:
            history_for_model = prices[:-1] if len(prices) > 1 else prices

        if current_price > 0.95 or current_price < 0.05:
            continue

        # ── Strategy 1: Naive ──
        if current_price < 0.50:
            size = min(2.0, results["naive"]["bankroll"] * 0.04)
            if size >= 0.50 and results["naive"]["bankroll"] >= size:
                trade = Trade(
                    market_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    direction="BUY",
                    entry_price=current_price,
                    exit_price=resolution,
                    size_usd=size,
                    edge_at_entry=0.5 - current_price,
                    strategy="naive",
                )
                trade.compute_pnl()
                results["naive"]["trades"].append(trade)
                results["naive"]["bankroll"] += trade.pnl
                results["naive"]["peak"] = max(results["naive"]["peak"], results["naive"]["bankroll"])

        # ── Strategy 2: Markov Only ──
        estimate = markov.estimate(
            market.condition_id, history_for_model, current_price, horizon_steps=20
        )
        markov_edge = estimate.raw_probability - current_price
        if markov_edge > 0.03 and estimate.confidence > 0.3:
            size = kelly_size(markov_edge, current_price, results["markov_only"]["bankroll"])
            if size >= 0.50 and results["markov_only"]["bankroll"] >= size:
                direction = "BUY"
                trade = Trade(
                    market_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    direction=direction,
                    entry_price=current_price,
                    exit_price=resolution,
                    size_usd=size,
                    edge_at_entry=markov_edge,
                    strategy="markov_only",
                )
                trade.compute_pnl()
                results["markov_only"]["trades"].append(trade)
                results["markov_only"]["bankroll"] += trade.pnl
                results["markov_only"]["peak"] = max(results["markov_only"]["peak"], results["markov_only"]["bankroll"])
        elif markov_edge < -0.03 and estimate.confidence > 0.3:
            # Sell YES / Buy NO
            size = kelly_size(abs(markov_edge), 1.0 - current_price, results["markov_only"]["bankroll"])
            if size >= 0.50 and results["markov_only"]["bankroll"] >= size:
                trade = Trade(
                    market_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    direction="SELL",
                    entry_price=current_price,
                    exit_price=resolution,
                    size_usd=size,
                    edge_at_entry=abs(markov_edge),
                    strategy="markov_only",
                )
                trade.compute_pnl()
                results["markov_only"]["trades"].append(trade)
                results["markov_only"]["bankroll"] += trade.pnl
                results["markov_only"]["peak"] = max(results["markov_only"]["peak"], results["markov_only"]["bankroll"])

        # ── Strategy 3: Markov + Calibration ──
        cal_estimate = markov.estimate(
            market.condition_id + "_cal", history_for_model, current_price,
            horizon_steps=20, calibrator=calibrator
        )
        cal_edge = cal_estimate.calibrated_probability - current_price
        cat_mult = calibrator.get_category_multiplier(market.category)

        # Apply NO-side premium: if model says sell YES, boost edge
        no_bonus = 0.0
        if cal_edge < 0:
            no_bonus = calibrator.get_no_side_edge(current_price) * 0.3

        adjusted_edge = cal_edge - no_bonus if cal_edge < 0 else cal_edge

        if abs(adjusted_edge) > 0.03 and cal_estimate.confidence > 0.3:
            direction = "BUY" if adjusted_edge > 0 else "SELL"
            edge_mag = abs(adjusted_edge) * cat_mult
            if direction == "BUY":
                size = kelly_size(edge_mag, current_price, results["markov_calibrated"]["bankroll"])
            else:
                size = kelly_size(edge_mag, 1.0 - current_price, results["markov_calibrated"]["bankroll"])

            if size >= 0.50 and results["markov_calibrated"]["bankroll"] >= size:
                trade = Trade(
                    market_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    direction=direction,
                    entry_price=current_price,
                    exit_price=resolution,
                    size_usd=size,
                    edge_at_entry=edge_mag,
                    strategy="markov_calibrated",
                )
                trade.compute_pnl()
                results["markov_calibrated"]["trades"].append(trade)
                results["markov_calibrated"]["bankroll"] += trade.pnl
                results["markov_calibrated"]["peak"] = max(results["markov_calibrated"]["peak"], results["markov_calibrated"]["bankroll"])

        # ── Strategy 4: Full System ──
        signals = {}

        # Signal 1: Markov MC
        if abs(markov_edge) > 0.02:
            signals["markov_mc"] = {
                "direction": "BUY" if markov_edge > 0 else "SELL",
                "edge": abs(markov_edge),
                "strength": estimate.confidence,
            }
            combiner.record_prediction("markov_mc", abs(markov_edge), "BUY" if markov_edge > 0 else "SELL")

        # Signal 2: Markov Absorbing
        absorb_edge = estimate.metadata.get("absorbing_prob", current_price) - current_price
        if abs(absorb_edge) > 0.02:
            signals["markov_absorb"] = {
                "direction": "BUY" if absorb_edge > 0 else "SELL",
                "edge": abs(absorb_edge),
                "strength": estimate.confidence * 0.9,
            }
            combiner.record_prediction("markov_absorb", abs(absorb_edge), "BUY" if absorb_edge > 0 else "SELL")

        # Signal 3: Implied probability (YES+NO != 1)
        if len(market.outcome_prices) >= 2:
            yes_p = market.outcome_prices[0] if market.outcome_prices[0] < 1 else current_price
            # Use current price as proxy
            vig = (current_price + (1 - current_price)) - 1.0  # Should be ~0 in liquid markets
            struct_edge = -vig / 2  # Half the overpricing is our edge
            if abs(struct_edge) > 0.01:
                signals["implied_prob"] = {
                    "direction": "BUY" if struct_edge > 0 else "SELL",
                    "edge": abs(struct_edge),
                    "strength": 0.5,
                }

        # Signal 4: NO-side bias
        no_side_edge = calibrator.get_no_side_edge(current_price)
        if no_side_edge > 0.03:
            signals["no_bias"] = {
                "direction": "SELL",  # Buy NO
                "edge": no_side_edge * 0.5,
                "strength": 0.6,
            }
            combiner.record_prediction("no_bias", no_side_edge * 0.5, "SELL")

        # Signal 5: Maker edge (always positive for limit orders)
        signals["maker_edge"] = {
            "direction": "BUY" if markov_edge > 0 else "SELL" if markov_edge < 0 else "BUY",
            "edge": 0.0112,  # Constant maker edge
            "strength": 0.4,
        }

        # Combine signals
        combined = combiner.combine_signals(signals)
        if combined and combined.combined_edge > 0.02:
            direction = combined.direction
            final_edge = combined.combined_edge * cat_mult

            if direction == "BUY":
                size = kelly_size(final_edge, current_price, results["full_system"]["bankroll"])
            else:
                size = kelly_size(final_edge, 1.0 - current_price, results["full_system"]["bankroll"])

            if size >= 0.50 and results["full_system"]["bankroll"] >= size:
                trade = Trade(
                    market_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    direction=direction,
                    entry_price=current_price,
                    exit_price=resolution,
                    size_usd=size,
                    edge_at_entry=final_edge,
                    strategy="full_system",
                )
                trade.compute_pnl()
                results["full_system"]["trades"].append(trade)
                results["full_system"]["bankroll"] += trade.pnl
                results["full_system"]["peak"] = max(results["full_system"]["peak"], results["full_system"]["bankroll"])

                # Record outcomes for IC learning
                for sig_name in signals:
                    combiner.record_outcome(sig_name, trade.return_pct)

    return results


# ─── Reporting ─────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], starting_capital: float = 50.0) -> dict:
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "total_pnl": 0, "avg_return": 0,
            "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
            "final_bankroll": starting_capital, "total_return_pct": 0,
            "avg_edge": 0, "best_trade": 0, "worst_trade": 0,
            "category_breakdown": {},
        }

    returns = [t.return_pct for t in trades]
    pnls = [t.pnl for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(pnls)
    equity = [starting_capital]
    for p in pnls:
        equity.append(equity[-1] + p)

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = (peak - eq_arr) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(drawdowns))

    returns_arr = np.array(returns)
    sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 1e-10 else 0.0

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Category breakdown
    categories = {}
    for t in trades:
        cat = t.category
        if cat not in categories:
            categories[cat] = {"trades": 0, "wins": 0, "pnl": 0.0}
        categories[cat]["trades"] += 1
        if t.pnl > 0:
            categories[cat]["wins"] += 1
        categories[cat]["pnl"] += t.pnl

    for cat in categories:
        n = categories[cat]["trades"]
        categories[cat]["win_rate"] = categories[cat]["wins"] / n if n > 0 else 0

    # Direction breakdown
    buy_trades = [t for t in trades if t.direction == "BUY"]
    sell_trades = [t for t in trades if t.direction == "SELL"]
    buy_wr = len([t for t in buy_trades if t.pnl > 0]) / len(buy_trades) if buy_trades else 0
    sell_wr = len([t for t in sell_trades if t.pnl > 0]) / len(sell_trades) if sell_trades else 0

    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(float(np.mean(returns_arr)) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "final_bankroll": round(starting_capital + total_pnl, 2),
        "total_return_pct": round(total_pnl / starting_capital * 100, 2),
        "avg_edge": round(float(np.mean([t.edge_at_entry for t in trades])) * 100, 2),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "n_buys": len(buy_trades),
        "n_sells": len(sell_trades),
        "buy_win_rate": round(buy_wr * 100, 1),
        "sell_win_rate": round(sell_wr * 100, 1),
        "category_breakdown": categories,
    }


def print_results(all_results: dict, combiner: AlphaCombiner | None = None):
    """Print formatted backtest results."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 BACKTEST — RESULTS")
    logger.info("=" * 70)

    # Summary table
    header = f"{'Strategy':<25} {'Trades':>7} {'Win%':>7} {'PnL':>9} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'PF':>7}"
    logger.info(f"\n{header}")
    logger.info("-" * 85)

    all_metrics = {}
    for name, data in all_results.items():
        metrics = compute_metrics(data["trades"])
        all_metrics[name] = metrics
        row = (
            f"{name:<25} "
            f"{metrics['n_trades']:>7d} "
            f"{metrics['win_rate']*100:>6.1f}% "
            f"${metrics['total_pnl']:>8.2f} "
            f"{metrics['total_return_pct']:>8.1f}% "
            f"{metrics['sharpe']:>8.2f} "
            f"{metrics['max_drawdown']:>7.1f}% "
            f"{metrics['profit_factor']:>7.2f}"
        )
        logger.info(row)

    # Detailed per-strategy breakdown
    for name, data in all_results.items():
        metrics = all_metrics[name]
        if metrics["n_trades"] == 0:
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  {name.upper()} — DETAILED BREAKDOWN")
        logger.info(f"{'─' * 50}")
        logger.info(f"  Starting Capital:  $50.00")
        logger.info(f"  Final Bankroll:    ${metrics['final_bankroll']:.2f}")
        logger.info(f"  Total P&L:         ${metrics['total_pnl']:+.2f}")
        logger.info(f"  Total Return:      {metrics['total_return_pct']:+.1f}%")
        logger.info(f"  Trades:            {metrics['n_trades']} ({metrics['n_buys']} buys, {metrics['n_sells']} sells)")
        logger.info(f"  Win Rate:          {metrics['win_rate']*100:.1f}%")
        logger.info(f"  Buy Win Rate:      {metrics['buy_win_rate']:.1f}%")
        logger.info(f"  Sell Win Rate:     {metrics['sell_win_rate']:.1f}%")
        logger.info(f"  Avg Return/Trade:  {metrics['avg_return']:+.2f}%")
        logger.info(f"  Avg Edge at Entry: {metrics['avg_edge']:.2f}%")
        logger.info(f"  Sharpe Ratio:      {metrics['sharpe']:.2f}")
        logger.info(f"  Max Drawdown:      {metrics['max_drawdown']:.1f}%")
        logger.info(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
        logger.info(f"  Best Trade:        ${metrics['best_trade']:+.2f}")
        logger.info(f"  Worst Trade:       ${metrics['worst_trade']:+.2f}")

        # Category breakdown
        if metrics["category_breakdown"]:
            logger.info(f"\n  Category Breakdown:")
            logger.info(f"  {'Category':<15} {'Trades':>7} {'Win%':>7} {'PnL':>9}")
            for cat, stats in sorted(metrics["category_breakdown"].items(), key=lambda x: -x[1]["pnl"]):
                wr = stats["win_rate"] * 100
                logger.info(f"  {cat:<15} {stats['trades']:>7} {wr:>6.1f}% ${stats['pnl']:>8.2f}")

        # Sample trades
        top_winners = sorted(data["trades"], key=lambda t: t.pnl, reverse=True)[:3]
        top_losers = sorted(data["trades"], key=lambda t: t.pnl)[:3]

        if top_winners:
            logger.info(f"\n  Top Winners:")
            for t in top_winners:
                logger.info(f"    {t.direction} @ ${t.entry_price:.2f} → ${t.exit_price:.0f} | "
                           f"PnL: ${t.pnl:+.2f} ({t.return_pct*100:+.1f}%) | {t.question[:50]}")

        if top_losers:
            logger.info(f"\n  Top Losers:")
            for t in top_losers:
                logger.info(f"    {t.direction} @ ${t.entry_price:.2f} → ${t.exit_price:.0f} | "
                           f"PnL: ${t.pnl:+.2f} ({t.return_pct*100:+.1f}%) | {t.question[:50]}")

    # Alpha Combination Engine stats
    if combiner:
        stats = combiner.get_fundamental_law_stats()
        logger.info(f"\n{'=' * 70}")
        logger.info("ALPHA COMBINATION ENGINE — FUNDAMENTAL LAW STATS")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Average IC:        {stats['avg_ic']:.4f}")
        logger.info(f"  Effective N:       {stats['effective_n']:.2f}")
        logger.info(f"  Information Ratio: {stats['information_ratio']:.4f}")
        logger.info(f"  Strategies tracked: {stats['n_strategies_tracked']}")
        logger.info(f"  Strategies w/ IC:  {stats['n_strategies_with_ic']}")

        if stats["per_strategy_ic"]:
            logger.info(f"\n  Per-Strategy IC:")
            for name, ic in sorted(stats["per_strategy_ic"].items(), key=lambda x: -x[1]):
                logger.info(f"    {name:<20} IC: {ic:+.4f}")

        if stats["optimal_weights"]:
            logger.info(f"\n  Optimal Weights:")
            for name, w in sorted(stats["optimal_weights"].items(), key=lambda x: -x[1]):
                logger.info(f"    {name:<20} {w:.4f}")

    logger.info(f"\n{'=' * 70}")
    logger.info("BACKTEST COMPLETE")
    logger.info(f"{'=' * 70}\n")

    return all_metrics


# ─── Main ──────────────────────────────────────────────────────────

async def main():
    start_time = time.time()

    # Collect data
    markets = await collect_backtest_data(max_markets=200)

    if not markets:
        logger.error("No market data collected. Check API connectivity.")
        return

    # Run backtest
    results = run_backtest(markets)

    # Create combiner for stats (reuse would be better but this shows the concept)
    combiner = AlphaCombiner(min_trades_for_ic=10)
    for name in ["markov_mc", "markov_absorb", "implied_prob", "no_bias", "maker_edge"]:
        combiner.register_strategy(name)
    # Feed full_system trades into combiner for stats
    for trade in results["full_system"]["trades"]:
        for sig in ["markov_mc", "markov_absorb", "no_bias", "maker_edge"]:
            combiner.record_prediction(sig, trade.edge_at_entry, trade.direction)
            combiner.record_outcome(sig, trade.return_pct)

    # Print results
    print_results(results, combiner)

    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
