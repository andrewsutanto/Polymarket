#!/usr/bin/env python3
"""Live Paper Trader — real-time Polymarket trading simulation.

Connects to live Polymarket APIs (Gamma + CLOB), runs the Markov model
in real-time, and simulates trades with realistic latency and fill modeling.

No real money is risked. All trades are logged to SQLite for analysis.

Usage:
    python live_paper_trader.py [--interval 120] [--capital 50]

Features:
    - Polls Gamma API for market universe discovery
    - Polls CLOB for real-time order books
    - Runs Markov Chain model on price history
    - Applies bias calibration + NO-side premium
    - Simulates maker limit order fills with latency
    - Logs all trades and portfolio state to SQLite
    - Prints live dashboard to terminal
    - Telegram notifications (if configured)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp")
    sys.exit(1)

from core.markov_model import MarkovModel
from core.bias_calibrator import BiasCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


# ─── Data Models ───────────────────────────────────────────────────

@dataclass
class LiveMarket:
    condition_id: str
    question: str
    slug: str
    category: str
    outcomes: list[str]
    token_ids: list[str]
    liquidity: float
    volume_24h: float
    end_date: str
    yes_price: float = 0.0
    no_price: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    price_history: list[float] = field(default_factory=list)
    last_update: float = 0.0


@dataclass
class PaperTrade:
    timestamp: str
    market_id: str
    question: str
    category: str
    direction: str
    entry_price: float
    size_usd: float
    edge: float
    status: str = "OPEN"  # OPEN, FILLED, EXPIRED
    exit_price: float = 0.0
    pnl: float = 0.0
    fill_delay_ms: int = 0


@dataclass
class RiskConfig:
    """Risk rules inspired by LobeHub paper trader."""
    max_position_pct: float = 0.10       # Max 10% of portfolio per trade
    max_drawdown_pct: float = 0.30       # Stop trading at 30% drawdown
    max_concurrent_positions: int = 5    # Max 5 open positions
    daily_loss_limit_pct: float = 0.05   # 5% daily loss limit
    max_market_exposure_pct: float = 0.20  # Max 20% in single market
    high_conviction_only_pct: float = 0.30  # Only high-conviction when cash < 30%


@dataclass
class Portfolio:
    starting_capital: float
    cash: float
    positions: dict = field(default_factory=dict)  # token_id -> {size, entry_price, direction}
    total_trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    peak_value: float = 0.0
    daily_pnl: float = 0.0
    daily_start_value: float = 0.0
    risk: RiskConfig = field(default_factory=RiskConfig)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_trades, 1)

    @property
    def value(self) -> float:
        pos_value = sum(p.get("size", 0) for p in self.positions.values())
        return self.cash + pos_value

    @property
    def return_pct(self) -> float:
        return (self.value - self.starting_capital) / self.starting_capital * 100

    @property
    def drawdown_pct(self) -> float:
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.value) / self.peak_value

    def check_risk(self, size_usd: float, edge: float) -> tuple[bool, str]:
        """Validate trade against risk rules. Returns (allowed, reason)."""
        # Max drawdown kill switch
        if self.drawdown_pct >= self.risk.max_drawdown_pct:
            return False, f"drawdown {self.drawdown_pct:.1%} >= {self.risk.max_drawdown_pct:.0%} limit"

        # Max concurrent positions
        if len(self.positions) >= self.risk.max_concurrent_positions:
            return False, f"max {self.risk.max_concurrent_positions} concurrent positions"

        # Daily loss limit
        if self.daily_start_value > 0:
            daily_loss = (self.daily_start_value - self.value) / self.daily_start_value
            if daily_loss >= self.risk.daily_loss_limit_pct:
                return False, f"daily loss {daily_loss:.1%} >= {self.risk.daily_loss_limit_pct:.0%} limit"

        # Max position size (% of portfolio)
        max_size = self.value * self.risk.max_position_pct
        if size_usd > max_size:
            return False, f"size ${size_usd:.2f} > {self.risk.max_position_pct:.0%} of portfolio (${max_size:.2f})"

        # High-conviction only mode (when cash is low)
        cash_ratio = self.cash / max(self.starting_capital, 0.01)
        if cash_ratio < self.risk.high_conviction_only_pct and edge < 0.06:
            return False, f"cash {cash_ratio:.0%} < {self.risk.high_conviction_only_pct:.0%}, need edge > 6%"

        return True, "ok"


# ─── Database ──────────────────────────────────────────────────────

def init_db(path: str = "data/paper_trades.db") -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_id TEXT,
            question TEXT,
            category TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            size_usd REAL,
            edge REAL,
            pnl REAL,
            status TEXT,
            fill_delay_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash REAL,
            positions_value REAL,
            total_value REAL,
            total_trades INTEGER,
            win_rate REAL,
            total_pnl REAL
        )
    """)
    conn.commit()
    return conn


def log_trade(conn: sqlite3.Connection, trade: PaperTrade):
    conn.execute(
        "INSERT INTO trades (timestamp, market_id, question, category, direction, "
        "entry_price, exit_price, size_usd, edge, pnl, status, fill_delay_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade.timestamp, trade.market_id, trade.question, trade.category,
         trade.direction, trade.entry_price, trade.exit_price, trade.size_usd,
         trade.edge, trade.pnl, trade.status, trade.fill_delay_ms),
    )
    conn.commit()


def log_snapshot(conn: sqlite3.Connection, portfolio: Portfolio):
    conn.execute(
        "INSERT INTO portfolio_snapshots (timestamp, cash, positions_value, "
        "total_value, total_trades, win_rate, total_pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), portfolio.cash,
         portfolio.value - portfolio.cash, portfolio.value,
         portfolio.total_trades, portfolio.win_rate, portfolio.total_pnl),
    )
    conn.commit()


# ─── Market Data ───────────────────────────────────────────────────

def classify(text: str, tags: list[str]) -> str:
    combined = f"{text} {' '.join(tags)}".lower()
    for cat, kws in {
        "sports": ["nba", "nfl", "mlb", "nhl", "ufc", "tennis", "game", "match", "set 1"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "xrp", "dogecoin"],
        "politics": ["president", "election", "trump", "biden", "congress", "vote"],
        "macro": ["fed", "interest rate", "inflation", "gdp", "tariff"],
    }.items():
        if any(kw in combined for kw in kws):
            return cat
    return "other"


def is_coinflip(question: str) -> bool:
    q = question.lower()
    return ("up or down" in q and any(x in q for x in ["am", "pm", "et"])) or \
           ("penta kill" in q) or ("odd/even" in q)


async def fetch_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch active markets from Gamma API."""
    try:
        params = {"active": "true", "closed": "false", "limit": 100,
                  "order": "volume24hr", "ascending": "false"}
        async with session.get(
            f"{GAMMA_API}/markets", params=params,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        logger.warning(f"Gamma API error: {e}")
    return []


async def fetch_book(session: aiohttp.ClientSession, token_id: str) -> dict:
    """Fetch order book for a token."""
    try:
        async with session.get(
            f"{CLOB_API}/book", params={"token_id": token_id},
            timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return {}


async def fetch_price_history(session: aiohttp.ClientSession, token_id: str) -> list[float]:
    """Fetch price history for Markov model."""
    try:
        async with session.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": "60"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [h["p"] for h in data.get("history", [])]
    except Exception:
        pass
    return []


# ─── Order Book Walking (realistic fills) ────────────────────────

def walk_book(
    levels: list[dict],
    size_usd: float,
    side: str = "buy",
) -> tuple[float, float]:
    """Simulate filling an order by walking the order book.

    Instead of using mid-price + flat slippage, this walks through
    actual order book levels to compute a realistic VWAP fill price.

    Args:
        levels: List of {"price": str, "size": str} from CLOB API.
               For buy orders, use asks (ascending price).
               For sell orders, use bids (descending price).
        size_usd: Dollar amount to fill.
        side: "buy" or "sell".

    Returns:
        (vwap_fill_price, filled_usd) — may be partially filled.
    """
    if not levels:
        return 0.0, 0.0

    total_cost = 0.0
    total_shares = 0.0
    remaining_usd = size_usd

    for level in levels:
        price = float(level.get("price", 0))
        size = float(level.get("size", 0))
        if price <= 0 or size <= 0:
            continue

        level_usd = price * size  # total USD at this level
        take_usd = min(remaining_usd, level_usd)
        take_shares = take_usd / price

        total_cost += take_usd
        total_shares += take_shares
        remaining_usd -= take_usd

        if remaining_usd <= 0.001:
            break

    if total_shares <= 0:
        return 0.0, 0.0

    vwap = total_cost / total_shares
    return round(vwap, 6), round(total_cost, 4)


# ─── Trading Logic ─────────────────────────────────────────────────

def kelly_size(edge: float, price: float, bankroll: float) -> float:
    if price <= 0.01 or price >= 0.99 or edge <= 0:
        return 0.0
    odds = (1.0 / price) - 1.0
    if odds <= 0:
        return 0.0
    p = min(price + edge, 0.99)
    q = 1.0 - p
    kelly = (p * odds - q) / odds * 0.25  # Quarter-Kelly
    if kelly <= 0:
        return 0.0
    return max(0.50, min(kelly * bankroll, 5.0))


async def evaluate_market(
    market: LiveMarket,
    markov: MarkovModel,
    calibrator: BiasCalibrator,
    portfolio: Portfolio,
) -> PaperTrade | None:
    """Evaluate a market for trading opportunity."""

    # Skip coin-flip markets
    if is_coinflip(market.question):
        return None

    # Need enough price history for Markov
    if len(market.price_history) < 15:
        return None

    # Skip if already have position
    for tid in market.token_ids:
        if tid in portfolio.positions:
            return None

    current_price = market.mid_price
    if current_price < 0.08 or current_price > 0.92:
        return None

    # Run Markov model
    estimate = markov.estimate(
        market.condition_id,
        market.price_history,
        current_price,
        horizon_steps=20,
        calibrator=calibrator,
    )

    if estimate.confidence < 0.3:
        return None

    cal_edge = estimate.calibrated_probability - current_price
    cat_mult = calibrator.get_category_multiplier(market.category)
    no_edge = calibrator.get_no_side_edge(current_price)

    # Combine signals
    MAKER_EDGE = 0.0112
    if cal_edge < 0:
        total_edge = abs(cal_edge) * cat_mult + no_edge * 0.4 + MAKER_EDGE
        direction = "SELL"
        price_for_kelly = 1.0 - current_price
    elif cal_edge > 0:
        total_edge = cal_edge * cat_mult * 0.7 + MAKER_EDGE  # Discount YES
        direction = "BUY"
        price_for_kelly = current_price
    else:
        return None

    if total_edge < 0.035:
        return None

    # Volatility filter
    if len(market.price_history) >= 20:
        vol = np.std(market.price_history[-20:])
        if vol > 0.25:
            return None

    size = kelly_size(total_edge, price_for_kelly, portfolio.cash)
    if size < 0.50 or portfolio.cash < size:
        return None

    # Simulate fill delay (100-500ms like real API)
    fill_delay = np.random.randint(100, 500)

    return PaperTrade(
        timestamp=datetime.now(timezone.utc).isoformat(),
        market_id=market.condition_id,
        question=market.question,
        category=market.category,
        direction=direction,
        entry_price=current_price,
        size_usd=round(size, 2),
        edge=round(total_edge, 4),
        fill_delay_ms=fill_delay,
    )


# ─── Main Loop ─────────────────────────────────────────────────────

async def run(interval: int = 120, capital: float = 50.0):
    logger.info("=" * 60)
    logger.info("  LIVE PAPER TRADER — Polymarket")
    logger.info("=" * 60)
    logger.info(f"  Starting capital: ${capital:.2f}")
    logger.info(f"  Scan interval:    {interval}s")
    logger.info(f"  Mode:             PAPER (no real money)")
    logger.info(f"  Model:            Markov Chain + Bias Calibration")
    logger.info(f"  Kelly fraction:   0.25 (quarter)")
    logger.info("=" * 60)

    markov = MarkovModel(n_states=10, n_simulations=5000)
    calibrator = BiasCalibrator()
    portfolio = Portfolio(
        starting_capital=capital, cash=capital,
        peak_value=capital, daily_start_value=capital,
    )
    db = init_db()

    tracked_markets: dict[str, LiveMarket] = {}
    cycle = 0

    async with aiohttp.ClientSession() as session:
        while True:
            cycle += 1
            t0 = time.time()
            logger.info(f"\n{'─' * 50}")
            logger.info(f"  SCAN CYCLE #{cycle}")
            logger.info(f"{'─' * 50}")

            # 1. Discover markets
            raw_markets = await fetch_markets(session)
            new_count = 0

            for raw in raw_markets:
                cid = raw.get("conditionId", "")
                if not cid:
                    continue

                tokens_raw = raw.get("clobTokenIds", "[]")
                if isinstance(tokens_raw, str):
                    try:
                        token_ids = json.loads(tokens_raw)
                    except:
                        continue
                elif isinstance(tokens_raw, list):
                    token_ids = tokens_raw
                else:
                    continue
                if not token_ids:
                    continue

                outcomes = raw.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except:
                        outcomes = ["Yes", "No"]

                price_str = raw.get("outcomePrices", "[]")
                if isinstance(price_str, str):
                    try:
                        prices = [float(p) for p in json.loads(price_str)]
                    except:
                        prices = []
                elif isinstance(price_str, list):
                    prices = [float(p) for p in price_str]
                else:
                    prices = []

                tags = raw.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = []

                question = raw.get("question", "")[:100]
                category = classify(f"{question} {raw.get('description', '')}", [str(t) for t in (tags or [])])

                if cid not in tracked_markets:
                    new_count += 1

                lm = LiveMarket(
                    condition_id=cid,
                    question=question,
                    slug=raw.get("slug", ""),
                    category=category,
                    outcomes=outcomes,
                    token_ids=[str(t) for t in token_ids],
                    liquidity=float(raw.get("liquidity", 0) or 0),
                    volume_24h=float(raw.get("volume24hr", 0) or 0),
                    end_date=raw.get("endDate", ""),
                    yes_price=prices[0] if prices else 0,
                    no_price=prices[1] if len(prices) > 1 else 0,
                    mid_price=prices[0] if prices else 0,
                )
                # Preserve existing history
                if cid in tracked_markets:
                    lm.price_history = tracked_markets[cid].price_history
                tracked_markets[cid] = lm

            logger.info(f"  Markets tracked: {len(tracked_markets)} ({new_count} new)")

            # 2. Fetch price histories for markets missing them
            markets_needing_history = [
                m for m in tracked_markets.values()
                if len(m.price_history) < 15 and m.token_ids
            ]
            fetched = 0
            for m in markets_needing_history[:20]:  # Cap per cycle
                history = await fetch_price_history(session, m.token_ids[0])
                if history:
                    m.price_history = history
                    fetched += 1
                await asyncio.sleep(0.15)

            if fetched:
                logger.info(f"  Fetched history for {fetched} markets")

            # 3. Update current prices from books
            top_markets = sorted(
                tracked_markets.values(),
                key=lambda m: m.volume_24h, reverse=True
            )[:30]

            for m in top_markets:
                if not m.token_ids:
                    continue
                book = await fetch_book(session, m.token_ids[0])
                if book:
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    best_bid = float(bids[0].get("price", 0)) if bids else 0
                    best_ask = float(asks[0].get("price", 1)) if asks else 1
                    m.mid_price = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else m.yes_price
                    m.spread = best_ask - best_bid
                    # Append to history
                    if m.mid_price > 0:
                        m.price_history.append(m.mid_price)
                await asyncio.sleep(0.1)

            # 4. Evaluate opportunities (with risk checks + book walking)
            trades_this_cycle = 0
            for m in top_markets:
                trade = await evaluate_market(m, markov, calibrator, portfolio)
                if trade and trades_this_cycle < 3:  # Max 3 trades per cycle
                    # Risk check
                    allowed, reason = portfolio.check_risk(trade.size_usd, trade.edge)
                    if not allowed:
                        logger.info(f"  RISK BLOCKED: {reason} | {m.question[:40]}")
                        continue

                    # Simulate fill delay
                    await asyncio.sleep(trade.fill_delay_ms / 1000.0)

                    # Order book walking for realistic fill price
                    book = await fetch_book(session, m.token_ids[0])
                    if book:
                        if trade.direction == "BUY":
                            asks = book.get("asks", [])
                            fill_price, filled_usd = walk_book(asks, trade.size_usd, "buy")
                        else:
                            bids = book.get("bids", [])
                            fill_price, filled_usd = walk_book(bids, trade.size_usd, "sell")

                        if fill_price > 0 and filled_usd > 0:
                            trade.entry_price = fill_price
                            trade.size_usd = min(trade.size_usd, filled_usd)
                        else:
                            logger.info(f"  NO FILL: empty book for {m.question[:40]}")
                            continue
                    else:
                        # Fallback: use mid-price with 0.5% slippage
                        if trade.direction == "BUY":
                            trade.entry_price = round(trade.entry_price * 1.005, 4)
                        else:
                            trade.entry_price = round(trade.entry_price * 0.995, 4)

                    trade.status = "FILLED"

                    # Update portfolio
                    portfolio.cash -= trade.size_usd
                    token_id = m.token_ids[0] if m.token_ids else m.condition_id
                    portfolio.positions[token_id] = {
                        "size": trade.size_usd,
                        "entry_price": trade.entry_price,
                        "direction": trade.direction,
                        "market_id": m.condition_id,
                        "question": m.question,
                    }
                    portfolio.total_trades += 1
                    trades_this_cycle += 1

                    log_trade(db, trade)

                    logger.info(
                        f"  TRADE: {trade.direction} {m.question[:50]}"
                        f"\n           @ {trade.entry_price:.4f} | ${trade.size_usd:.2f} | "
                        f"edge: {trade.edge*100:.1f}% | delay: {trade.fill_delay_ms}ms"
                    )

            # 5. Check for resolved markets (positions to close)
            for token_id, pos in list(portfolio.positions.items()):
                # Check if market has resolved
                cid = pos.get("market_id", "")
                if cid in tracked_markets:
                    m = tracked_markets[cid]
                    # If price is very close to 0 or 1, likely resolved
                    if m.mid_price >= 0.98 or m.mid_price <= 0.02:
                        resolution = 1.0 if m.mid_price >= 0.98 else 0.0
                        entry = pos["entry_price"]
                        size = pos["size"]

                        if pos["direction"] == "BUY":
                            shares = size / entry if entry > 0 else 0
                            pnl = shares * resolution - size
                        else:
                            no_price = 1.0 - entry
                            shares = size / no_price if no_price > 0 else 0
                            pnl = shares * (1.0 - resolution) - size

                        # Add maker bonus
                        pnl += size * 0.0112

                        portfolio.cash += size + pnl
                        portfolio.total_pnl += pnl
                        if pnl > 0:
                            portfolio.wins += 1
                        del portfolio.positions[token_id]

                        logger.info(
                            f"  ✅ RESOLVED: {pos['question'][:50]}"
                            f"\n              PnL: ${pnl:+.2f} | WR: {portfolio.win_rate*100:.0f}%"
                        )

            # 6. Portfolio snapshot
            portfolio.peak_value = max(portfolio.peak_value, portfolio.value)
            log_snapshot(db, portfolio)

            elapsed = time.time() - t0

            # 7. Dashboard
            logger.info(f"\n  ┌──────────────────────────────────────┐")
            logger.info(f"  │  PORTFOLIO STATUS                     │")
            logger.info(f"  ├──────────────────────────────────────┤")
            logger.info(f"  │  Cash:         ${portfolio.cash:>8.2f}             │")
            logger.info(f"  │  Positions:    {len(portfolio.positions):>3}                    │")
            logger.info(f"  │  Total Value:  ${portfolio.value:>8.2f}             │")
            logger.info(f"  │  Total P&L:    ${portfolio.total_pnl:>+8.2f}             │")
            logger.info(f"  │  Return:       {portfolio.return_pct:>+7.1f}%              │")
            logger.info(f"  │  Trades:       {portfolio.total_trades:>4}                   │")
            logger.info(f"  │  Win Rate:     {portfolio.win_rate*100:>5.1f}%                │")
            logger.info(f"  │  Cycle time:   {elapsed:>5.1f}s                │")
            logger.info(f"  └──────────────────────────────────────┘")

            logger.info(f"\n  Next scan in {interval}s... (Ctrl+C to stop)")
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Live Paper Trader for Polymarket")
    parser.add_argument("--interval", type=int, default=120, help="Scan interval in seconds")
    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital in USD")
    args = parser.parse_args()

    try:
        asyncio.run(run(interval=args.interval, capital=args.capital))
    except KeyboardInterrupt:
        logger.info("\n\nPaper trader stopped. Trades logged to data/paper_trades.db")


if __name__ == "__main__":
    main()
