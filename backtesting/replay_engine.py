#!/usr/bin/env python3
"""Time-traveling replay engine for PMXT orderbook data.

Reconstructs market state at each point in time from PMXT Parquet snapshots,
then feeds it to the execution agent as if it were live. The execution agent
has NO access to future data — look-ahead bias is eliminated by construction.

Architecture:
    REPLAY ENGINE                    EXECUTION AGENT
    ─────────────                    ───────────────
    Loads parquet file               Receives market state
    Maintains internal clock         as if it were LIVE
    At each tick:                    Makes trade decisions
      → Serves orderbook            Has NO access to future
      → Serves price history         data or the replay DB
      → Simulates fills
      → Advances clock              Returns: BUY/SELL/HOLD

Usage:
    python backtesting/replay_engine.py --file data/pmxt_orderbooks/polymarket_orderbook_2026-04-08T06.parquet
    python backtesting/replay_engine.py --file data/pmxt_orderbooks/polymarket_orderbook_2026-04-08T06.parquet --market-index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.markov_model import MarkovModel
from core.bias_calibrator import BiasCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────

@dataclass
class OrderbookState:
    """Current state of an orderbook at a point in time."""
    token_id: str = ""
    best_bid: float = 0.0
    best_ask: float = 1.0
    mid_price: float = 0.5
    spread: float = 1.0
    last_update: float = 0.0
    side: str = ""  # YES or NO


@dataclass
class MarketState:
    """Full state of a market visible to the execution agent."""
    market_id: str
    token_ids: list[str] = field(default_factory=list)
    books: dict[str, OrderbookState] = field(default_factory=dict)  # token_id -> book
    price_history: list[float] = field(default_factory=list)  # mid prices over time
    update_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0


@dataclass
class ReplayTrade:
    """A simulated trade from the replay."""
    timestamp: float
    market_id: str
    token_id: str
    direction: str  # BUY or SELL
    entry_price: float
    size_usd: float
    edge: float
    exit_price: float = 0.0
    pnl: float = 0.0
    resolved: bool = False
    eval_latency_ms: float = 0.0  # How long our model took to decide
    price_at_eval_start: float = 0.0  # Price when we started evaluating
    latency_slippage: float = 0.0  # Price moved during our evaluation


# ─── Replay Engine ────────────────────────────────────────────────

class ReplayEngine:
    """Replays PMXT orderbook data chronologically.

    Reads a Parquet file in chunks, reconstructs orderbook state
    at each timestamp, and exposes it via get_market_state().
    """

    def __init__(self, parquet_path: str | list[str], chunk_size: int = 50_000):
        """Initialize with one or multiple parquet files (loaded in order)."""
        if isinstance(parquet_path, str):
            self.paths = [parquet_path]
        else:
            self.paths = sorted(parquet_path)  # Sort by filename = chronological
        self.chunk_size = chunk_size
        self.markets: dict[str, MarketState] = {}
        self._events: list[dict] = []
        self._current_idx = 0
        self._clock = 0.0

    def load_events(self, max_rows: int | None = None) -> int:
        """Load and parse events from one or more Parquet files in chronological order."""
        import pyarrow.parquet as pq

        logger.info("Loading %d file(s)...", len(self.paths))
        t0 = time.time()

        all_events = []
        total_rows = 0

        for fpath in self.paths:
            logger.info("  Reading %s", Path(fpath).name)
            pf = pq.ParquetFile(fpath)

            for rg_idx in range(pf.metadata.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                df = rg.to_pandas()
                total_rows += len(df)

                for _, row in df.iterrows():
                    try:
                        data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
                        ts = row["timestamp_received"]
                        if hasattr(ts, "timestamp"):
                            ts_float = ts.timestamp()
                        else:
                            ts_float = float(ts)

                        all_events.append({
                            "timestamp": ts_float,
                            "market_id": row["market_id"],
                            "update_type": row["update_type"],
                            "data": data,
                        })
                    except Exception:
                        continue

                if max_rows and total_rows >= max_rows:
                    break

            logger.info("    -> %d events so far", len(all_events))
            if max_rows and total_rows >= max_rows:
                break

        # Sort by timestamp
        all_events.sort(key=lambda e: e["timestamp"])
        self._events = all_events

        elapsed = time.time() - t0
        logger.info("Loaded %d events from %d files in %.1fs", len(self._events), len(self.paths), elapsed)
        return len(self._events)

    def advance_to(self, target_ts: float) -> int:
        """Process all events up to target_ts. Returns number processed."""
        processed = 0
        while self._current_idx < len(self._events):
            evt = self._events[self._current_idx]
            if evt["timestamp"] > target_ts:
                break

            self._process_event(evt)
            self._current_idx += 1
            self._clock = evt["timestamp"]
            processed += 1

        return processed

    def advance_n(self, n: int) -> int:
        """Process the next N events. Returns number processed."""
        processed = 0
        end_idx = min(self._current_idx + n, len(self._events))

        while self._current_idx < end_idx:
            evt = self._events[self._current_idx]
            self._process_event(evt)
            self._current_idx += 1
            self._clock = evt["timestamp"]
            processed += 1

        return processed

    def _process_event(self, evt: dict) -> None:
        """Update market state from a single event."""
        mid = evt["market_id"]
        data = evt["data"]
        ts = evt["timestamp"]

        if mid not in self.markets:
            self.markets[mid] = MarketState(
                market_id=mid,
                first_seen=ts,
            )

        market = self.markets[mid]
        market.last_seen = ts
        market.update_count += 1

        # Extract token info
        token_id = data.get("token_id", "")
        if token_id and token_id not in market.token_ids:
            market.token_ids.append(token_id)

        # Update orderbook
        if token_id:
            if token_id not in market.books:
                market.books[token_id] = OrderbookState(token_id=token_id)

            book = market.books[token_id]
            book.side = data.get("side", book.side)
            book.last_update = ts

            try:
                bb = data.get("best_bid")
                if bb is not None:
                    book.best_bid = float(bb)
                ba = data.get("best_ask")
                if ba is not None:
                    book.best_ask = float(ba)
            except (TypeError, ValueError):
                pass

            if book.best_bid > 0 and book.best_ask > 0:
                book.mid_price = (book.best_bid + book.best_ask) / 2
                book.spread = book.best_ask - book.best_bid

            # Track price history (sample every ~10 updates to keep manageable)
            if market.update_count % 10 == 0 and book.mid_price > 0:
                market.price_history.append(book.mid_price)

    def get_market_state(self, market_id: str) -> MarketState | None:
        """Get current state of a market (what the execution agent sees)."""
        return self.markets.get(market_id)

    def get_active_markets(self, min_updates: int = 50) -> list[MarketState]:
        """Get all markets with enough data to trade."""
        return [
            m for m in self.markets.values()
            if m.update_count >= min_updates and len(m.price_history) >= 15
        ]

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def progress(self) -> float:
        if not self._events:
            return 0.0
        return self._current_idx / len(self._events)

    @property
    def events_remaining(self) -> int:
        return len(self._events) - self._current_idx


# ─── Execution Agent (isolated — NO access to future data) ────────

class ExecutionAgent:
    """Makes trading decisions based ONLY on current market state.

    This agent is intentionally isolated from the replay engine.
    It receives market snapshots and returns trade decisions.
    It cannot see future events or the full dataset.
    """

    def __init__(self, capital: float = 1000.0):
        self.capital = capital
        self.cash = capital
        self.positions: dict[str, dict] = {}
        self.trades: list[ReplayTrade] = []
        self.markov = MarkovModel(n_states=10, n_simulations=2000)
        self.calibrator = BiasCalibrator()

    def evaluate(self, market: MarketState) -> ReplayTrade | None:
        """Evaluate a market for trading. Returns trade or None.

        This function receives ONLY the current market state.
        It has NO access to future data, the replay engine, or the parquet file.
        """
        if not market.price_history or len(market.price_history) < 15:
            return None

        # Get the YES-side book
        yes_book = None
        for tid, book in market.books.items():
            if book.side == "YES" or (not yes_book and book.mid_price > 0):
                yes_book = book

        if not yes_book or yes_book.mid_price < 0.08 or yes_book.mid_price > 0.92:
            return None

        if yes_book.spread > 0.10:
            return None  # Too wide, likely stale

        # Skip if already have position in this market
        if market.market_id in self.positions:
            return None

        current_price = yes_book.mid_price

        # Run Markov model on available price history
        estimate = self.markov.estimate(
            market.market_id,
            market.price_history,
            current_price,
            horizon_steps=20,
            calibrator=self.calibrator,
        )

        if estimate.confidence < 0.3:
            return None

        # Compute edge
        cal_edge = estimate.calibrated_probability - current_price
        cat_mult = 1.2  # Default category multiplier
        no_edge = self.calibrator.get_no_side_edge(current_price)

        if cal_edge < 0:
            total_edge = abs(cal_edge) * cat_mult + no_edge * 0.4
            direction = "SELL"
            price_for_kelly = 1.0 - current_price
        elif cal_edge > 0:
            total_edge = cal_edge * cat_mult * 0.7
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

        # Kelly sizing (quarter-Kelly, max 10% of portfolio)
        if price_for_kelly <= 0.01 or price_for_kelly >= 0.99:
            return None
        odds = (1.0 / price_for_kelly) - 1.0
        if odds <= 0:
            return None
        p = min(price_for_kelly + total_edge, 0.99)
        kelly = ((p * odds - (1 - p)) / odds) * 0.25
        if kelly <= 0:
            return None
        size = max(0.50, min(kelly * self.cash, self.cash * 0.10))

        if size < 0.50 or self.cash < size:
            return None

        # Apply spread as slippage
        if direction == "BUY":
            fill_price = yes_book.best_ask  # Buy at ask
        else:
            fill_price = yes_book.best_bid  # Sell at bid

        if fill_price <= 0:
            fill_price = current_price

        return ReplayTrade(
            timestamp=market.last_seen,
            market_id=market.market_id,
            token_id=yes_book.token_id,
            direction=direction,
            entry_price=fill_price,
            size_usd=round(size, 2),
            edge=round(total_edge, 4),
        )

    def execute_trade(self, trade: ReplayTrade) -> None:
        """Execute a trade (update portfolio)."""
        self.cash -= trade.size_usd
        self.positions[trade.market_id] = {
            "trade": trade,
            "entry_price": trade.entry_price,
            "size": trade.size_usd,
            "direction": trade.direction,
        }
        self.trades.append(trade)

    def check_exits(self, markets: dict[str, MarketState]) -> list[ReplayTrade]:
        """Check if any positions should be closed."""
        closed = []
        for mid, pos in list(self.positions.items()):
            market = markets.get(mid)
            if not market:
                continue

            # Check if price moved to resolution (0 or 1)
            for book in market.books.values():
                if book.mid_price >= 0.97 or book.mid_price <= 0.03:
                    resolution = 1.0 if book.mid_price >= 0.97 else 0.0
                    trade = pos["trade"]

                    if trade.direction == "BUY":
                        shares = trade.size_usd / trade.entry_price
                        pnl = shares * resolution - trade.size_usd
                    else:
                        no_price = 1.0 - trade.entry_price
                        shares = trade.size_usd / no_price if no_price > 0 else 0
                        pnl = shares * (1.0 - resolution) - trade.size_usd

                    pnl -= trade.size_usd * 0.02  # Taker fee

                    trade.exit_price = resolution
                    trade.pnl = round(pnl, 4)
                    trade.resolved = True

                    self.cash += trade.size_usd + pnl
                    closed.append(trade)
                    del self.positions[mid]
                    break

        return closed

    def get_stats(self) -> dict[str, Any]:
        """Get portfolio statistics."""
        resolved = [t for t in self.trades if t.resolved]
        if not resolved:
            return {
                "total_trades": len(self.trades),
                "resolved": 0,
                "open_positions": len(self.positions),
                "cash": round(self.cash, 2),
                "total_value": round(self.cash + sum(p["size"] for p in self.positions.values()), 2),
            }

        wins = sum(1 for t in resolved if t.pnl > 0)
        pnls = [t.pnl for t in resolved]

        return {
            "total_trades": len(self.trades),
            "resolved": len(resolved),
            "open_positions": len(self.positions),
            "wins": wins,
            "win_rate": round(wins / len(resolved) * 100, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(np.mean(pnls), 2),
            "sharpe": round(np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)), 2) if np.std(pnls) > 0 else 0,
            "max_drawdown": round(self._max_drawdown(pnls), 1),
            "cash": round(self.cash, 2),
            "return_pct": round((self.cash - self.capital) / self.capital * 100, 1),
        }

    def _max_drawdown(self, pnls: list[float]) -> float:
        equity = [self.capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd


# ─── Main Replay Loop ────────────────────────────────────────────

def run_replay(
    parquet_path: str,
    max_rows: int | None = None,
    capital: float = 1000.0,
    eval_every_n: int = 5000,
    realtime: bool = False,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Run the full replay simulation.

    Args:
        parquet_path: Path to PMXT Parquet file
        max_rows: Limit rows to process (None = all)
        capital: Starting capital
        eval_every_n: Evaluate markets every N events
        realtime: If True, pace events to match actual timestamps (shows latency)
        speed: Playback speed multiplier (1.0=realtime, 10.0=10x, 0=no delay)
    """
    engine = ReplayEngine(parquet_path)
    agent = ExecutionAgent(capital=capital)

    # Load events
    n_events = engine.load_events(max_rows=max_rows)
    if n_events == 0:
        return {"error": "No events loaded"}

    mode_str = f"REALTIME {speed}x" if realtime else "FAST (no delay)"
    logger.info("Starting replay: %d events, $%.0f capital, mode=%s", n_events, capital, mode_str)
    logger.info("=" * 60)

    t0 = time.time()
    eval_count = 0
    trade_count = 0
    last_event_ts = 0.0  # For realtime pacing
    wall_start = time.time()
    sim_start = 0.0  # Will be set on first event

    # Process in batches
    while engine.events_remaining > 0:
        # Realtime pacing: sleep to match actual event timing
        if realtime and speed > 0 and engine._current_idx < len(engine._events):
            evt_ts = engine._events[engine._current_idx]["timestamp"]
            if sim_start == 0.0:
                sim_start = evt_ts
                wall_start = time.time()
            else:
                # How far ahead in sim time are we?
                sim_elapsed = evt_ts - sim_start
                wall_elapsed = time.time() - wall_start
                target_wall = sim_elapsed / speed
                sleep_time = target_wall - wall_elapsed
                if sleep_time > 0.01:
                    time.sleep(min(sleep_time, 2.0))  # Cap at 2s to stay responsive

        # Advance by eval_every_n events
        processed = engine.advance_n(eval_every_n)
        eval_count += 1

        # Check exits first
        closed = agent.check_exits(engine.markets)
        for trade in closed:
            logger.info(
                "  CLOSED: %s %s | PnL: $%+.2f | Price: %.3f -> %.3f",
                trade.direction, trade.market_id[:16], trade.pnl,
                trade.entry_price, trade.exit_price,
            )

        # Evaluate active markets for new trades
        active = engine.get_active_markets(min_updates=50)
        for market in active[:20]:  # Cap evaluations per cycle
            # Capture price BEFORE evaluation (for latency slippage)
            pre_eval_prices = {}
            for tid, book in market.books.items():
                pre_eval_prices[tid] = book.mid_price

            eval_start = time.time()
            trade = agent.evaluate(market)
            eval_latency_ms = (time.time() - eval_start) * 1000

            if trade and trade_count < 50:  # Max trades per replay
                # Compute latency slippage: how much price moved during eval
                pre_price = pre_eval_prices.get(trade.token_id, trade.entry_price)
                trade.eval_latency_ms = eval_latency_ms
                trade.price_at_eval_start = pre_price
                trade.latency_slippage = abs(trade.entry_price - pre_price)

                agent.execute_trade(trade)
                trade_count += 1

                from datetime import datetime as dt_, timezone as tz_
                sim_time = dt_.fromtimestamp(engine.clock, tz=tz_.utc).strftime("%H:%M:%S")
                slip_bps = trade.latency_slippage * 10000
                logger.info(
                    "  [%s] TRADE #%d: %s %s | $%.2f @ %.4f | edge: %.1f%% | latency: %.0fms | slip: %.0fbps",
                    sim_time, trade_count, trade.direction, trade.market_id[:16],
                    trade.size_usd, trade.entry_price, trade.edge * 100,
                    eval_latency_ms, slip_bps,
                )

        # Progress update every 10 evals
        if eval_count % 10 == 0:
            pct = engine.progress * 100
            logger.info(
                "  [%.1f%%] %d events | %d markets | %d trades | $%.0f cash",
                pct, engine._current_idx, len(engine.markets),
                len(agent.trades), agent.cash,
            )

    elapsed = time.time() - t0

    # Final report
    stats = agent.get_stats()
    stats["runtime_s"] = round(elapsed, 1)
    stats["events_processed"] = n_events
    stats["markets_seen"] = len(engine.markets)
    stats["active_markets"] = len(engine.get_active_markets())

    # Latency stats
    if agent.trades:
        latencies = [t.eval_latency_ms for t in agent.trades]
        slippages = [t.latency_slippage * 10000 for t in agent.trades]  # in bps
        stats["avg_latency_ms"] = round(np.mean(latencies), 0)
        stats["max_latency_ms"] = round(max(latencies), 0)
        stats["avg_slippage_bps"] = round(np.mean(slippages), 1)
        stats["max_slippage_bps"] = round(max(slippages), 1)

    print("\n" + "=" * 60)
    print("  REPLAY RESULTS")
    print("=" * 60)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {v}")
    print("=" * 60)

    # Print individual trades
    if agent.trades:
        print(f"\n  TRADE LOG ({len(agent.trades)} trades):")
        print(f"  {'#':<4} {'Dir':<5} {'Market':<18} {'Entry':>8} {'Exit':>8} {'PnL':>10} {'Edge':>7}")
        print(f"  {'-'*65}")
        for i, t in enumerate(agent.trades):
            status = f"${t.pnl:+.2f}" if t.resolved else "OPEN"
            print(
                f"  {i+1:<4} {t.direction:<5} {t.market_id[:16]:<18} "
                f"{t.entry_price:>8.4f} {t.exit_price:>8.4f} {status:>10} "
                f"{t.edge*100:>6.1f}%"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(description="PMXT Replay Engine")
    parser.add_argument("--file", nargs="+", required=True, help="Path(s) to PMXT Parquet file(s)")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows (None=all)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital")
    parser.add_argument("--eval-every", type=int, default=5000, help="Evaluate every N events")
    parser.add_argument("--realtime", action="store_true", help="Pace to match actual timestamps")
    parser.add_argument("--speed", type=float, default=60.0, help="Playback speed (1=realtime, 60=1min/sec)")
    args = parser.parse_args()

    # Support glob patterns
    import glob as glob_mod
    files = []
    for pattern in args.file:
        expanded = glob_mod.glob(pattern)
        files.extend(expanded if expanded else [pattern])
    files = sorted(set(files))

    stats = run_replay(
        parquet_path=files if len(files) > 1 else files[0],
        max_rows=args.max_rows,
        capital=args.capital,
        eval_every_n=args.eval_every,
        realtime=args.realtime,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
