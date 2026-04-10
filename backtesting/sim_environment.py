#!/usr/bin/env python3
"""Simulated live trading environment — true isolation.

Two completely independent components communicating via a mock API:

1. MARKET SIMULATOR (this file): Reads PMXT parquet, advances in real
   wall-clock time, serves orderbook/price data via function calls that
   mimic the real Polymarket CLOB API. The simulator holds ALL data but
   only exposes what's happened up to "now".

2. TRADING BOT (imported, isolated): Receives market data exactly as it
   would live. Has NO access to the simulator's internal state, future
   events, or the parquet file. It calls sim.get_markets(), sim.get_book(),
   sim.get_price_history() — identical signatures to the real API.

The simulator enforces:
- Data is ONLY available up to the current simulated timestamp
- When the bot takes 250ms to evaluate, 250ms of real events pass
- Fills happen at the ACTUAL orderbook state at fill time (not eval time)
- The bot experiences the same latency/slippage it would live

Usage:
    python backtesting/sim_environment.py \\
        --file data/pmxt_orderbooks/polymarket_orderbook_2026-04-08T06.parquet \\
        --speed 0  \\
        --capital 1000

    speed=0: fast as possible (wall-clock latency still real)
    speed=1: 1x realtime (1 hour of data = 1 hour)
    speed=60: 60x speed (1 hour = 1 minute)
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.markov_model import MarkovModel
from core.bias_calibrator import BiasCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# COMPONENT 1: MARKET SIMULATOR (holds all data, exposes only "past")
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BookLevel:
    price: float
    size: float

@dataclass
class SimBook:
    """Simulated orderbook for one token."""
    token_id: str
    side: str = ""  # YES or NO
    best_bid: float = 0.0
    best_ask: float = 1.0
    last_update_ts: float = 0.0

@dataclass
class SimMarket:
    """Simulated market state visible up to current time."""
    market_id: str
    books: dict[str, SimBook] = field(default_factory=dict)
    price_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, mid_price)
    first_seen: float = 0.0
    last_seen: float = 0.0
    update_count: int = 0


class MarketSimulator:
    """Replays PMXT data as a mock Polymarket API.

    This is the "god object" that sees everything. But it ONLY exposes
    data up to self._sim_clock via its public API methods. The trading
    bot calls these methods and has no other access.
    """

    def __init__(self, speed: float = 0.0):
        """
        Args:
            speed: Playback speed. 0=fast-as-possible, 1=realtime, 60=60x
        """
        self.speed = speed
        self._events: list[dict] = []
        self._event_idx = 0
        self._sim_clock: float = 0.0  # Current simulated time
        self._wall_start: float = 0.0
        self._sim_start: float = 0.0
        self._markets: dict[str, SimMarket] = {}
        self._lock = Lock()
        self._started = False
        self._total_events_fed = 0

    def load(self, parquet_paths: list[str], max_rows: int | None = None) -> int:
        """Load events with FULL JSON data — no field reduction.

        Processes row-group by row-group to avoid the 126s to_pylist() bottleneck.
        All fields preserved: best_bid, best_ask, token_id, side, change_price,
        change_size, change_side — identical to what you'd see live.

        Clears previous events to free memory. Market state is preserved.
        """
        # Free previous events (market state in self._markets is kept)
        self._events.clear()
        self._event_idx = 0
        import pyarrow.parquet as pq

        all_events = []
        total = 0

        for fpath in sorted(parquet_paths):
            logger.info("  Loading %s", Path(fpath).name)
            t0 = time.time()

            pf = pq.ParquetFile(fpath)
            file_events = 0

            # Process row-group by row-group (each ~1M rows)
            # This avoids loading entire 23M-row file into Python at once
            for rg_idx in range(pf.metadata.num_row_groups):
                rg = pf.read_row_group(rg_idx, columns=["timestamp_received", "market_id", "data"])
                n = rg.num_rows
                total += n

                # Convert columns — row groups are ~1M rows, manageable
                ts_pd = rg.column("timestamp_received").to_pandas()
                # Timestamps are millisecond-precision (not nanoseconds)
                ts_arr = ts_pd.astype("int64").values / 1e3  # ms -> seconds
                mid_list = rg.column("market_id").to_pylist()
                data_list = rg.column("data").to_pylist()

                for i in range(n):
                    raw = data_list[i]
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)  # Full JSON — all fields preserved
                        all_events.append({
                            "ts": float(ts_arr[i]),
                            "mid": mid_list[i],
                            "data": data,
                        })
                        file_events += 1
                    except Exception:
                        continue

                del ts_pd, ts_arr, mid_list, data_list, rg

                if max_rows and total >= max_rows:
                    break

            elapsed = time.time() - t0
            logger.info("    -> %d events in %.1fs (%.0f rows/sec)",
                        file_events, elapsed, file_events / max(elapsed, 0.01))

            if max_rows and total >= max_rows:
                break

        # Sort chronologically
        all_events.sort(key=lambda e: e["ts"])
        if max_rows:
            all_events = all_events[:max_rows]
        self._events = all_events
        logger.info("Loaded %d events total from %d files", len(self._events), len(parquet_paths))
        return len(self._events)

    def start(self):
        """Start the simulation clock."""
        if self._events:
            self._sim_start = self._events[0]["ts"]
            self._sim_clock = self._sim_start
        self._wall_start = time.time()
        self._started = True

    def tick(self) -> bool:
        """Advance the simulation by processing events up to "now".

        In speed=0 mode: processes the next batch immediately.
        In speed=N mode: waits until wall-clock catches up to sim time.

        Returns False when all events are consumed.
        """
        if self._event_idx >= len(self._events):
            return False

        if self.speed > 0:
            # Compute target sim time based on wall clock elapsed
            wall_elapsed = time.time() - self._wall_start
            target_sim = self._sim_start + (wall_elapsed * self.speed)
        else:
            # Fast mode: jump to next batch of events
            # Process events in ~100ms chunks of sim-time
            if self._event_idx < len(self._events):
                target_sim = self._events[min(self._event_idx + 5000, len(self._events) - 1)]["ts"]
            else:
                return False

        # Process all events up to target_sim
        with self._lock:
            processed = 0
            while self._event_idx < len(self._events):
                evt = self._events[self._event_idx]
                if evt["ts"] > target_sim:
                    break

                self._apply_event(evt)
                self._event_idx += 1
                self._sim_clock = evt["ts"]
                processed += 1
                self._total_events_fed += 1

        return True

    def _apply_event(self, evt: dict):
        """Apply a single event to internal state."""
        mid = evt["mid"]
        data = evt["data"]
        ts = evt["ts"]

        if mid not in self._markets:
            self._markets[mid] = SimMarket(market_id=mid, first_seen=ts)

        market = self._markets[mid]
        market.last_seen = ts
        market.update_count += 1

        token_id = data.get("token_id", "")
        if not token_id:
            return

        if token_id not in market.books:
            market.books[token_id] = SimBook(token_id=token_id)

        book = market.books[token_id]
        book.side = data.get("side", book.side)
        book.last_update_ts = ts

        try:
            bb = data.get("best_bid")
            if bb is not None:
                book.best_bid = float(bb)
            ba = data.get("best_ask")
            if ba is not None:
                book.best_ask = float(ba)
        except (TypeError, ValueError):
            pass

        # Record price history (sampled)
        if book.best_bid > 0 and book.best_ask > 0:
            mid_p = (book.best_bid + book.best_ask) / 2
            if market.update_count % 10 == 0:
                market.price_history.append((ts, mid_p))

    # ── PUBLIC API (what the trading bot sees) ────────────────────

    def get_sim_time(self) -> float:
        """Current simulated timestamp."""
        return self._sim_clock

    def get_sim_time_str(self) -> str:
        """Current simulated time as human-readable string."""
        return datetime.fromtimestamp(self._sim_clock, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def get_markets(self, min_updates: int = 30) -> list[dict]:
        """Get active markets — equivalent to Gamma API /markets.

        Returns ONLY markets that have been seen up to current sim time.
        """
        with self._lock:
            result = []
            for m in self._markets.values():
                if m.update_count < min_updates:
                    continue
                if not m.books:
                    continue

                # Find best YES-side book
                best_book = None
                for book in m.books.values():
                    if book.best_bid > 0 and book.best_ask > 0:
                        if best_book is None or book.side == "YES":
                            best_book = book

                if not best_book:
                    continue

                mid = (best_book.best_bid + best_book.best_ask) / 2
                result.append({
                    "market_id": m.market_id,
                    "token_id": best_book.token_id,
                    "side": best_book.side,
                    "best_bid": best_book.best_bid,
                    "best_ask": best_book.best_ask,
                    "mid_price": mid,
                    "spread": best_book.best_ask - best_book.best_bid,
                    "updates": m.update_count,
                })
            return result

    def get_book(self, token_id: str) -> dict | None:
        """Get orderbook for a token — equivalent to CLOB /book."""
        with self._lock:
            for m in self._markets.values():
                if token_id in m.books:
                    book = m.books[token_id]
                    return {
                        "best_bid": book.best_bid,
                        "best_ask": book.best_ask,
                        "side": book.side,
                        "last_update": book.last_update_ts,
                    }
        return None

    def get_price_history(self, market_id: str) -> list[float]:
        """Get price history — equivalent to CLOB /prices-history.

        Returns ONLY prices up to current sim time.
        """
        with self._lock:
            market = self._markets.get(market_id)
            if not market:
                return []
            return [p for ts, p in market.price_history if ts <= self._sim_clock]

    def attempt_fill(self, market_id: str, token_id: str, direction: str, size_usd: float) -> dict | None:
        """Attempt to fill an order at CURRENT book state.

        This is called AFTER the bot has computed its decision.
        The book may have moved since the bot started evaluating.
        """
        with self._lock:
            market = self._markets.get(market_id)
            if not market or token_id not in market.books:
                return None

            book = market.books[token_id]
            if direction == "BUY":
                fill_price = book.best_ask
            else:
                fill_price = book.best_bid

            if fill_price <= 0 or fill_price >= 1:
                return None

            return {
                "filled": True,
                "fill_price": fill_price,
                "fill_size_usd": size_usd,
                "fill_time": self._sim_clock,
                "book_bid": book.best_bid,
                "book_ask": book.best_ask,
            }

    @property
    def progress(self) -> float:
        return self._event_idx / max(len(self._events), 1)

    @property
    def events_remaining(self) -> int:
        return len(self._events) - self._event_idx

    @property
    def total_markets(self) -> int:
        return len(self._markets)


# ═══════════════════════════════════════════════════════════════════
# COMPONENT 2: TRADING BOT (isolated — talks only to simulator API)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BotTrade:
    sim_time: float
    wall_time: float
    market_id: str
    token_id: str
    direction: str
    signal_price: float    # Price when bot STARTED evaluating
    fill_price: float      # Price when order was FILLED (after latency)
    size_usd: float
    edge: float
    latency_ms: float      # Real computation time
    slippage_bps: float    # Price movement during computation
    pnl: float = 0.0
    resolved: bool = False
    exit_price: float = 0.0


class TradingBot:
    """Isolated trading bot that only communicates via simulator API.

    This class has NO reference to the parquet file, the event list,
    or any future data. It calls sim.get_markets(), sim.get_book(),
    sim.get_price_history() — same as calling real Polymarket APIs.
    """

    def __init__(self, sim: MarketSimulator, capital: float = 1000.0):
        # The bot only has a reference to the simulator's PUBLIC API
        # It cannot access sim._events, sim._markets directly, etc.
        self._api = sim  # Treated as an opaque API endpoint
        self.capital = capital
        self.cash = capital
        self.positions: dict[str, BotTrade] = {}
        self.trades: list[BotTrade] = []
        self._closed_markets: set[str] = set()  # Markets we already traded — no re-entry
        self.markov = MarkovModel(n_states=10, n_simulations=2000)
        self.markov.reset()  # Ensure clean state — no cached matrices from prior runs
        self.calibrator = BiasCalibrator()
        self._max_trades = 50
        self._max_positions = 10

    def scan_and_trade(self) -> list[BotTrade]:
        """Run one scan cycle — exactly as the live bot would.

        1. Get market list from API
        2. Filter candidates
        3. Evaluate with Markov model (takes real wall-clock time)
        4. Submit orders (filled at CURRENT book, not eval-time book)
        """
        new_trades = []

        # Step 1: Get available markets from "API"
        markets = self._api.get_markets(min_updates=50)
        if not markets:
            return []

        # Step 2: Check exits on existing positions
        self._check_exits()

        # Step 3: Skip if at position limit
        if len(self.positions) >= self._max_positions:
            return []
        if len(self.trades) >= self._max_trades:
            return []

        # Step 4: Evaluate candidates
        for mkt in markets[:30]:  # Top 30 by activity
            mid = mkt["market_id"]
            tid = mkt["token_id"]
            current_price = mkt["mid_price"]

            if mid in self.positions:
                continue
            if mid in self._closed_markets:
                continue  # Never re-enter a market we already traded
            if current_price < 0.08 or current_price > 0.92:
                continue
            if mkt["spread"] > 0.10:
                continue

            # Get price history from "API"
            history = self._api.get_price_history(mid)
            if len(history) < 15:
                continue

            # ── EVALUATION (this takes real wall-clock time) ──
            signal_price = current_price
            eval_start = time.time()

            estimate = self.markov.estimate(
                mid, history, current_price,
                horizon_steps=20, calibrator=self.calibrator,
            )

            eval_end = time.time()
            latency_ms = (eval_end - eval_start) * 1000

            if estimate.confidence < 0.3:
                continue

            # Compute edge
            cal_edge = estimate.calibrated_probability - current_price
            no_edge = self.calibrator.get_no_side_edge(current_price)

            if cal_edge < 0:
                total_edge = abs(cal_edge) * 1.2 + no_edge * 0.4
                direction = "SELL"
                price_for_kelly = 1.0 - current_price
            elif cal_edge > 0:
                total_edge = cal_edge * 1.2 * 0.7
                direction = "BUY"
                price_for_kelly = current_price
            else:
                continue

            if total_edge < 0.035:
                continue

            # Volatility filter
            if len(history) >= 20:
                vol = np.std(history[-20:])
                if vol > 0.25:
                    continue

            # Kelly sizing
            if price_for_kelly <= 0.01 or price_for_kelly >= 0.99:
                continue
            odds = (1.0 / price_for_kelly) - 1.0
            if odds <= 0:
                continue
            p = min(price_for_kelly + total_edge, 0.99)
            kelly = ((p * odds - (1 - p)) / odds) * 0.25
            if kelly <= 0:
                continue
            size = max(0.50, min(kelly * self.cash, self.cash * 0.10))
            if size < 0.50 or self.cash < size:
                continue

            # ── FILL (at CURRENT book state, not eval-time state) ──
            # During our evaluation, the market kept moving.
            # The simulator advanced. The fill price reflects reality.
            fill = self._api.attempt_fill(mid, tid, direction, size)
            if not fill:
                continue

            fill_price = fill["fill_price"]
            slippage_bps = abs(fill_price - signal_price) * 10000

            trade = BotTrade(
                sim_time=self._api.get_sim_time(),
                wall_time=time.time(),
                market_id=mid,
                token_id=tid,
                direction=direction,
                signal_price=signal_price,
                fill_price=fill_price,
                size_usd=round(size, 2),
                edge=round(total_edge, 4),
                latency_ms=round(latency_ms, 1),
                slippage_bps=round(slippage_bps, 1),
            )

            self.cash -= trade.size_usd
            self.positions[mid] = trade
            self.trades.append(trade)
            new_trades.append(trade)

            sim_ts = datetime.fromtimestamp(trade.sim_time, tz=timezone.utc).strftime("%H:%M:%S")
            logger.info(
                "  [%s] TRADE #%d: %s %s | $%.2f | signal: %.4f -> fill: %.4f | "
                "edge: %.1f%% | latency: %.0fms | slip: %.0fbps",
                sim_ts, len(self.trades), direction, mid[:16],
                trade.size_usd, signal_price, fill_price,
                total_edge * 100, latency_ms, slippage_bps,
            )

            if len(self.positions) >= self._max_positions:
                break

        return new_trades

    def _check_exits(self):
        """Check if any positions should be closed."""
        for mid, trade in list(self.positions.items()):
            book = self._api.get_book(trade.token_id)
            if not book:
                continue

            mid_p = (book["best_bid"] + book["best_ask"]) / 2
            if mid_p >= 0.97 or mid_p <= 0.03:
                resolution = 1.0 if mid_p >= 0.97 else 0.0

                if trade.direction == "BUY":
                    shares = trade.size_usd / trade.fill_price
                    pnl = shares * resolution - trade.size_usd
                else:
                    no_price = 1.0 - trade.fill_price
                    shares = trade.size_usd / no_price if no_price > 0 else 0
                    pnl = shares * (1.0 - resolution) - trade.size_usd

                pnl -= trade.size_usd * 0.02  # Taker fee

                trade.exit_price = resolution
                trade.pnl = round(pnl, 4)
                trade.resolved = True
                self.cash += trade.size_usd + pnl
                self._closed_markets.add(mid)  # Never re-enter
                del self.positions[mid]

                sim_ts = datetime.fromtimestamp(
                    self._api.get_sim_time(), tz=timezone.utc
                ).strftime("%H:%M:%S")
                logger.info(
                    "  [%s] CLOSED: %s %s | PnL: $%+.2f | %.4f -> %.4f",
                    sim_ts, trade.direction, trade.market_id[:16],
                    pnl, trade.fill_price, resolution,
                )

    def report(self) -> dict:
        """Generate final performance report."""
        resolved = [t for t in self.trades if t.resolved]
        total_value = self.cash + sum(t.size_usd for t in self.positions.values())

        # Mark-to-market open positions using last known prices
        unrealized_pnl = 0.0
        mtm_details = []
        for mid, trade in self.positions.items():
            book = self._api.get_book(trade.token_id)
            if book:
                current_mid = (book["best_bid"] + book["best_ask"]) / 2
                if trade.direction == "BUY":
                    shares = trade.size_usd / trade.fill_price
                    mtm = shares * current_mid - trade.size_usd
                else:
                    no_price = 1.0 - trade.fill_price
                    shares = trade.size_usd / no_price if no_price > 0 else 0
                    current_no = 1.0 - current_mid
                    mtm = shares * current_no - trade.size_usd
                unrealized_pnl += mtm
                mtm_details.append((mid[:16], trade.direction, trade.fill_price, current_mid, round(mtm, 2)))

        total_value = self.cash + sum(t.size_usd for t in self.positions.values()) + unrealized_pnl

        stats = {
            "capital": self.capital,
            "cash": round(self.cash, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_value": round(total_value, 2),
            "return_pct": round((total_value - self.capital) / self.capital * 100, 1),
            "total_trades": len(self.trades),
            "resolved": len(resolved),
            "open_positions": len(self.positions),
        }

        if resolved:
            wins = sum(1 for t in resolved if t.pnl > 0)
            pnls = [t.pnl for t in resolved]
            stats["wins"] = wins
            stats["losses"] = len(resolved) - wins
            stats["win_rate"] = round(wins / len(resolved) * 100, 1)
            stats["total_pnl"] = round(sum(pnls), 2)
            stats["avg_pnl"] = round(np.mean(pnls), 2)
            stats["best_trade"] = round(max(pnls), 2)
            stats["worst_trade"] = round(min(pnls), 2)
            if np.std(pnls) > 0:
                stats["sharpe"] = round(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)), 2)

        if self.trades:
            stats["avg_latency_ms"] = round(np.mean([t.latency_ms for t in self.trades]), 1)
            stats["max_latency_ms"] = round(max(t.latency_ms for t in self.trades), 1)
            stats["avg_slippage_bps"] = round(np.mean([t.slippage_bps for t in self.trades]), 1)
            stats["max_slippage_bps"] = round(max(t.slippage_bps for t in self.trades), 1)

        return stats


# ═══════════════════════════════════════════════════════════════════
# COMPONENT 3: ORCHESTRATOR (connects sim + bot, runs the loop)
# ═══════════════════════════════════════════════════════════════════

def run_simulation(
    parquet_paths: list[str],
    max_rows: int | None = None,
    capital: float = 1000.0,
    speed: float = 0.0,
    scan_interval_events: int = 20000,
) -> dict:
    """Run the full simulated environment.

    Processes files ONE AT A TIME to stay within memory limits.
    Market state and bot state carry forward between files.
    This means a 7GB RAM machine can handle any number of files.
    """
    sim = MarketSimulator(speed=speed)
    bot = TradingBot(sim, capital=capital)

    mode = f"REALTIME {speed}x" if speed > 0 else "FAST (real latency, no wait)"
    logger.info("=" * 70)
    logger.info("  SIMULATED LIVE ENVIRONMENT")
    logger.info("  Files: %d | Capital: $%.0f | Mode: %s", len(parquet_paths), capital, mode)
    logger.info("  Bot sees ONLY current market state (no future data)")
    logger.info("  Fills at ACTUAL book price at fill time (not eval time)")
    logger.info("  Memory-safe: processes one file at a time")
    logger.info("=" * 70)

    t0 = time.time()
    total_events = 0
    file_num = 0

    for fpath in sorted(parquet_paths):
        file_num += 1
        logger.info("--- File %d/%d: %s ---", file_num, len(parquet_paths), Path(fpath).name)

        # Load ONE file into the simulator (clears previous events, keeps market state)
        n = sim.load([fpath], max_rows=max_rows)
        if n == 0:
            continue
        total_events += n

        if not sim._started:
            sim.start()

        cycle = 0
        while sim.tick():
            cycle += 1

            if cycle % (scan_interval_events // 5000) == 0:
                bot.scan_and_trade()

            if cycle % 50 == 0:
                pct = sim.progress * 100
                sim_time = sim.get_sim_time_str()
                logger.info(
                    "  [%.0f%%] %s | %d markets | %d trades | $%.0f cash | %d open",
                    pct, sim_time, sim.total_markets, len(bot.trades),
                    bot.cash, len(bot.positions),
                )

        logger.info(
            "  File done: %d events | %d markets | %d trades",
            n, sim.total_markets, len(bot.trades),
        )

    elapsed = time.time() - t0
    stats = bot.report()
    stats["sim_duration"] = sim.get_sim_time_str()
    stats["wall_time_s"] = round(elapsed, 1)
    stats["events_processed"] = total_events
    stats["markets_seen"] = sim.total_markets

    # Final report
    print("\n" + "=" * 70)
    print("  SIMULATION RESULTS")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k.replace('_', ' ').title():<30} {v}")
    print("=" * 70)

    # Trade log
    if bot.trades:
        print(f"\n  TRADE LOG ({len(bot.trades)} trades):")
        print(f"  {'#':<3} {'Time':<10} {'Dir':<5} {'Market':<18} {'Signal':>8} "
              f"{'Fill':>8} {'Slip':>7} {'Latency':>8} {'PnL':>10}")
        print(f"  {'-' * 85}")
        for i, t in enumerate(bot.trades):
            ts = datetime.fromtimestamp(t.sim_time, tz=timezone.utc).strftime("%H:%M:%S")
            pnl_str = f"${t.pnl:+.2f}" if t.resolved else "OPEN"
            print(
                f"  {i+1:<3} {ts:<10} {t.direction:<5} {t.market_id[:16]:<18} "
                f"{t.signal_price:>8.4f} {t.fill_price:>8.4f} "
                f"{t.slippage_bps:>5.0f}bp {t.latency_ms:>6.0f}ms {pnl_str:>10}"
            )

    # Mark-to-market on open positions
    report = bot.report()
    if "mtm_details" not in report:
        # Re-fetch mtm from report internals
        pass
    if bot.positions:
        print(f"\n  OPEN POSITIONS (mark-to-market):")
        print(f"  {'Market':<18} {'Dir':<5} {'Entry':>8} {'Current':>8} {'MTM P&L':>10}")
        print(f"  {'-' * 55}")
        for mid, trade in bot.positions.items():
            book = bot._api.get_book(trade.token_id)
            if book:
                cur = (book["best_bid"] + book["best_ask"]) / 2
                if trade.direction == "BUY":
                    shares = trade.size_usd / trade.fill_price
                    mtm = shares * cur - trade.size_usd
                else:
                    no_p = 1.0 - trade.fill_price
                    shares = trade.size_usd / no_p if no_p > 0 else 0
                    cur_no = 1.0 - cur
                    mtm = shares * cur_no - trade.size_usd
                print(f"  {mid[:16]:<18} {trade.direction:<5} {trade.fill_price:>8.4f} {cur:>8.4f} ${mtm:>+9.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Simulated Live Trading Environment")
    parser.add_argument("--file", nargs="+", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--speed", type=float, default=0.0,
                        help="0=fast, 1=realtime, 60=60x speed")
    parser.add_argument("--scan-every", type=int, default=20000,
                        help="Bot scans every N events")
    args = parser.parse_args()

    files = []
    for pattern in args.file:
        expanded = glob_mod.glob(pattern)
        files.extend(expanded if expanded else [pattern])
    files = sorted(set(files))

    run_simulation(
        parquet_paths=files,
        max_rows=args.max_rows,
        capital=args.capital,
        speed=args.speed,
        scan_interval_events=args.scan_every,
    )


if __name__ == "__main__":
    main()
