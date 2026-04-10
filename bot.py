#!/usr/bin/env python3
"""Polymarket Trading Bot — Telegram-controlled.

Full-featured Telegram bot for Polymarket prediction market trading.
Supports paper mode (shadow trading with realistic fills) and live mode.

Usage:
    # Set env vars first:
    #   TELEGRAM_BOT_TOKEN=your_token
    #   TELEGRAM_CHAT_ID=your_chat_id
    #   (Optional for live: POLYMARKET_PRIVATE_KEY, POLYMARKET_API_KEY, etc.)

    python bot.py --mode paper --capital 50
    python bot.py --mode live
    python bot.py --mode paper --ws   # Enable WebSocket feed

Telegram Commands:
    /start          — Welcome + mode info
    /status         — Portfolio overview
    /positions      — Open positions with mark-to-market
    /trades         — Recent trade history
    /pnl            — P&L breakdown
    /scan           — Trigger manual scan
    /opportunities  — Current signals ranked by edge
    /markets        — Tracked market universe
    /mode           — Toggle paper/live mode
    /capital <amt>  — Set starting capital (paper mode)
    /kill           — Emergency stop all trading
    /resume         — Resume trading after kill
    /stats          — Fundamental Law stats (IC, effective-N, IR)
    /weights        — Bayesian strategy weight posteriors
    /smartmoney     — Recent large order flow
    /config         — Show/edit strategy parameters
    /help           — Command reference
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import aiohttp
except ImportError:
    sys.exit("pip install aiohttp")

try:
    from telegram import (
        Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup,
    )
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler, ContextTypes,
    )
except ImportError:
    sys.exit("pip install python-telegram-bot")

from core.markov_model import MarkovModel
from core.bias_calibrator import BiasCalibrator
from core.alpha_combiner import AlphaCombiner
from core.state_manager import StateManager
from core.bayesian_updater import BayesianUpdater
from core.smart_money import SmartMoneyDetector
from core.ws_feed import WSFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


# ─── Order Book Walking (realistic fills) ────────────────────────

def walk_book(
    levels: list[dict],
    size_usd: float,
    side: str = "buy",
) -> tuple[float, float]:
    """Simulate filling an order by walking the order book.

    Instead of using mid-price + flat slippage, this walks through
    actual order book levels to compute a realistic VWAP fill price.

    Ported from live_paper_trader.py for realistic paper-trade fills.

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


# ─── Risk Rules (ported from live_paper_trader.py) ───────────────

MAX_POSITION_PCT = 0.10        # Max 10% of portfolio per trade
MAX_DRAWDOWN_PCT = 0.30        # Stop trading at 30% drawdown
MAX_CONCURRENT_POSITIONS = 5   # Max 5 open positions
DAILY_LOSS_LIMIT_PCT = 0.05   # 5% daily loss limit
HIGH_CONVICTION_EDGE = 0.06   # Min edge when cash is low
HIGH_CONVICTION_CASH_PCT = 0.30  # "Low cash" threshold


def check_risk(
    size_usd: float,
    edge: float,
    cash: float,
    total_value: float,
    starting_capital: float,
    peak_value: float,
    n_positions: int,
    daily_start_value: float,
) -> tuple[bool, str]:
    """Validate trade against risk rules. Returns (allowed, reason).

    Ported from live_paper_trader.py Portfolio.check_risk().
    """
    # Max drawdown kill switch
    if peak_value > 0:
        dd = (peak_value - total_value) / peak_value
        if dd >= MAX_DRAWDOWN_PCT:
            return False, f"drawdown {dd:.1%} >= {MAX_DRAWDOWN_PCT:.0%} limit"

    # Max concurrent positions
    if n_positions >= MAX_CONCURRENT_POSITIONS:
        return False, f"max {MAX_CONCURRENT_POSITIONS} concurrent positions"

    # Daily loss limit
    if daily_start_value > 0:
        daily_loss = (daily_start_value - total_value) / daily_start_value
        if daily_loss >= DAILY_LOSS_LIMIT_PCT:
            return False, f"daily loss {daily_loss:.1%} >= {DAILY_LOSS_LIMIT_PCT:.0%} limit"

    # Max position size (% of portfolio)
    max_size = total_value * MAX_POSITION_PCT
    if size_usd > max_size:
        return False, f"size ${size_usd:.2f} > {MAX_POSITION_PCT:.0%} of portfolio (${max_size:.2f})"

    # High-conviction only mode (when cash is low)
    cash_ratio = cash / max(starting_capital, 0.01)
    if cash_ratio < HIGH_CONVICTION_CASH_PCT and edge < HIGH_CONVICTION_EDGE:
        return False, f"cash {cash_ratio:.0%} < {HIGH_CONVICTION_CASH_PCT:.0%}, need edge > 6%"

    return True, "ok"


# ─── Data Models ───────────────────────────────────────────────────

@dataclass
class Market:
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
    best_bid: float = 0.0
    best_ask: float = 1.0
    price_history: list[float] = field(default_factory=list)
    # Structural fields for combinatorial arbitrage (from Gamma API)
    event_slug: str = ""           # Groups outcomes of same event
    neg_risk_market_id: str = ""   # NegRisk contract grouping ($28.8M source)
    group_item_title: str = ""     # UI grouping title


@dataclass
class PaperPosition:
    token_id: str
    market_id: str
    question: str
    category: str
    direction: str  # BUY (yes) or SELL (no)
    entry_price: float
    size_usd: float
    edge_at_entry: float
    limit_price: float  # The price our limit order was placed at
    filled: bool = False
    fill_time: str = ""
    opened_at: str = ""


@dataclass
class TradeRecord:
    timestamp: str
    market_id: str
    question: str
    category: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    edge: float
    pnl: float
    mode: str  # PAPER or LIVE


# ─── Bot State ─────────────────────────────────────────────────────

class BotState:
    """Central state for the trading bot."""

    def __init__(self, mode: str = "paper", capital: float = 50.0):
        self.mode = mode  # "paper" or "live"
        self.starting_capital = capital
        self.cash = capital
        self.positions: dict[str, PaperPosition] = {}
        self.trades: list[TradeRecord] = []
        self.markets: dict[str, Market] = {}
        self.signals: list[dict] = []  # Recent signals
        self.kill_switch = False
        self.scan_count = 0
        self.start_time = datetime.now(timezone.utc)

        # Risk tracking (ported from live_paper_trader.py)
        self.peak_value = capital
        self.daily_start_value = capital

        # Alerts
        self.alerts_enabled = False
        self.alerts_threshold = 5.0  # percent
        self.price_snapshots: dict[str, float] = {}  # condition_id -> last price

        # Models
        self.markov = MarkovModel(n_states=10, n_simulations=5000)
        self.calibrator = BiasCalibrator()
        self.combiner = AlphaCombiner(min_trades_for_ic=10)
        for name in ["markov_mc", "markov_absorb", "no_bias", "maker_edge"]:
            self.combiner.register_strategy(name)

        # Bayesian weight updater — learns from resolved trades
        self.bayesian_updater = BayesianUpdater(
            prior_alpha=1.0, prior_beta=1.0,
            ic_window=100, min_observations=10,
        )
        for name in ["markov_mc", "markov_absorb", "no_bias", "maker_edge"]:
            self.bayesian_updater.register_strategy(name)

        # Smart money detector — track large institutional orders
        self.smart_money = SmartMoneyDetector(
            large_order_threshold_usd=1000.0,
            flow_window_s=3600.0,
        )

        # WSFeed instance (started only when --ws flag is used)
        self.ws_feed: WSFeed | None = None

        # DB
        self.db = self._init_db()

        # Persistent state manager (crash recovery)
        self.state_mgr = StateManager(db_path="data/bot_state.db")
        self._recover_from_crash()

    def _recover_from_crash(self) -> None:
        """Recover portfolio state from SQLite after crash/restart."""
        try:
            cash, positions, trades = self.state_mgr.load_state()
            if cash is not None:
                self.cash = cash
                for tid, pos_data in positions.items():
                    self.positions[tid] = PaperPosition(
                        token_id=pos_data.get("token_id", tid),
                        market_id=pos_data.get("market_id", ""),
                        question=pos_data.get("question", ""),
                        category=pos_data.get("category", ""),
                        direction=pos_data.get("direction", "BUY"),
                        entry_price=pos_data.get("entry_price", 0.0),
                        size_usd=pos_data.get("size_usd", 0.0),
                        edge_at_entry=pos_data.get("edge_at_entry", 0.0),
                        limit_price=pos_data.get("limit_price", 0.0),
                        filled=pos_data.get("filled", False),
                        fill_time=pos_data.get("fill_time", ""),
                        opened_at=pos_data.get("opened_at", ""),
                    )
                logger.info(
                    "Crash recovery: $%.2f cash, %d positions",
                    self.cash, len(self.positions),
                )
        except Exception as e:
            logger.warning("No prior state to recover: %s", e)

    def persist_state(self) -> None:
        """Persist current portfolio state to SQLite (call after every trade)."""
        positions_data = {
            tid: {
                "token_id": p.token_id, "market_id": p.market_id,
                "question": p.question, "category": p.category,
                "direction": p.direction, "entry_price": p.entry_price,
                "size_usd": p.size_usd, "edge_at_entry": p.edge_at_entry,
                "limit_price": p.limit_price, "filled": p.filled,
                "opened_at": p.opened_at,
            }
            for tid, p in self.positions.items()
        }
        self.state_mgr.save_state(self.cash, positions_data, [])

    @property
    def total_value(self) -> float:
        pos_value = sum(p.size_usd for p in self.positions.values())
        return self.cash + pos_value

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len([t for t in self.trades if t.pnl > 0]) / len(self.trades)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def uptime(self) -> str:
        delta = datetime.now(timezone.utc) - self.start_time
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"

    def _init_db(self):
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/bot_trades.db")
        conn.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, market_id TEXT, question TEXT, category TEXT,
            direction TEXT, entry_price REAL, exit_price REAL,
            size_usd REAL, edge REAL, pnl REAL, mode TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, cash REAL, positions_count INTEGER,
            total_value REAL, total_pnl REAL, win_rate REAL
        )""")
        conn.commit()
        return conn

    def log_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.db.execute(
            "INSERT INTO trades (timestamp, market_id, question, category, direction, "
            "entry_price, exit_price, size_usd, edge, pnl, mode) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trade.timestamp, trade.market_id, trade.question, trade.category,
             trade.direction, trade.entry_price, trade.exit_price,
             trade.size_usd, trade.edge, trade.pnl, trade.mode),
        )
        self.db.commit()

    def snapshot(self):
        self.db.execute(
            "INSERT INTO snapshots (timestamp, cash, positions_count, total_value, total_pnl, win_rate) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), self.cash,
             len(self.positions), self.total_value, self.total_pnl, self.win_rate),
        )
        self.db.commit()


# ─── Market Data ───────────────────────────────────────────────────

def classify(text: str, tags: list[str]) -> str:
    combined = f"{text} {' '.join(tags)}".lower()
    for cat, kws in {
        "sports": ["nba", "nfl", "mlb", "nhl", "ufc", "tennis", "game", "match", "set 1", "kills"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto", "xrp", "dogecoin", "up or down"],
        "politics": ["president", "election", "trump", "biden", "congress", "vote", "pm ", "minister"],
        "macro": ["fed", "interest rate", "inflation", "gdp", "tariff"],
    }.items():
        if any(kw in combined for kw in kws):
            return cat
    return "other"


def is_coinflip(q: str) -> bool:
    q = q.lower()
    return ("up or down" in q and any(x in q for x in ["am", "pm", "et"])) or \
           ("penta kill" in q) or ("odd/even" in q)


async def fetch_with_backoff(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
    max_retries: int = 3,
    timeout_s: float = 15,
) -> dict | list | None:
    """Fetch with exponential backoff on 429/5xx errors.

    Addresses: rate limits, transient failures, API throttling.
    """
    for attempt in range(max_retries):
        try:
            async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    wait = 2 ** attempt + 1
                    logger.warning(f"Rate limited (429), waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                if resp.status >= 500:
                    await asyncio.sleep(1)
                    continue
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on {url} (attempt {attempt+1})")
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Fetch error {url}: {e}")
            return None
    return None


async def fetch_markets(session: aiohttp.ClientSession) -> list[dict]:
    result = await fetch_with_backoff(
        session, f"{GAMMA_API}/markets",
        params={"active": "true", "closed": "false", "limit": 100,
                "order": "volume24hr", "ascending": "false"},
    )
    return result if isinstance(result, list) else []


async def fetch_book(session: aiohttp.ClientSession, token_id: str) -> dict:
    result = await fetch_with_backoff(
        session, f"{CLOB_API}/book",
        params={"token_id": token_id}, timeout_s=8,
    )
    return result if isinstance(result, dict) else {}


async def fetch_history(session: aiohttp.ClientSession, token_id: str) -> list[float]:
    result = await fetch_with_backoff(
        session, f"{CLOB_API}/prices-history",
        params={"market": token_id, "interval": "max", "fidelity": "60"},
        timeout_s=10,
    )
    if isinstance(result, dict):
        return [h["p"] for h in result.get("history", [])]
    return []


# ─── Trading Logic ─────────────────────────────────────────────────

def kelly_size(edge: float, price: float, bankroll: float) -> float:
    if price <= 0.01 or price >= 0.99 or edge <= 0:
        return 0.0
    odds = (1.0 / price) - 1.0
    if odds <= 0:
        return 0.0
    p = min(price + edge, 0.99)
    kelly = ((p * odds - (1 - p)) / odds) * 0.25
    if kelly <= 0:
        return 0.0
    return max(0.50, min(kelly * bankroll, 5.0))


def evaluate_market(
    market: Market,
    state: BotState,
    smart_money_signal: dict | None = None,
) -> dict | None:
    """Evaluate a market and return signal dict or None.

    Args:
        market: Market data to evaluate.
        state: Current bot state.
        smart_money_signal: Optional smart money signal from SmartMoneyDetector.
            When present and agrees with direction, boosts edge.
            When present and disagrees, vetoes the trade.
    """
    if is_coinflip(market.question):
        return None
    if len(market.price_history) < 15:
        return None
    if market.mid_price < 0.08 or market.mid_price > 0.92:
        return None
    # Skip if already positioned
    for tid in market.token_ids:
        if tid in state.positions:
            return None

    price = market.mid_price
    est = state.markov.estimate(
        market.condition_id, market.price_history, price,
        horizon_steps=20, calibrator=state.calibrator,
    )
    if est.confidence < 0.3:
        return None

    cal_edge = est.calibrated_probability - price
    cat_mult = state.calibrator.get_category_multiplier(market.category)
    no_edge = state.calibrator.get_no_side_edge(price)
    # NOTE: Maker edge (+1.12%) removed. It was inflating perceived edge
    # without modeling fill probability. If we implement maker execution
    # with shadow fills, we can add it back — but only on confirmed fills.

    if cal_edge < 0:
        total_edge = abs(cal_edge) * cat_mult + no_edge * 0.4
        direction = "SELL"
        kelly_price = 1.0 - price
    elif cal_edge > 0:
        total_edge = cal_edge * cat_mult * 0.7
        direction = "BUY"
        kelly_price = price
    else:
        return None

    # --- Smart money confirmation / veto ---
    sm_confirms = False
    if smart_money_signal:
        sm_dir = smart_money_signal["direction"]
        sm_edge = smart_money_signal.get("edge", 0)
        sm_strength = smart_money_signal.get("strength", 0)
        if sm_dir == direction:
            # Smart money agrees — boost edge proportionally
            total_edge += sm_edge * sm_strength * 0.3
            sm_confirms = True
        else:
            # Smart money disagrees — veto the trade (strong disagreement)
            if sm_strength >= 0.6:
                return None
            # Mild disagreement — reduce edge
            total_edge *= 0.6

    if total_edge < 0.035:
        return None

    # Volatility filter
    if len(market.price_history) >= 20:
        vol = np.std(market.price_history[-20:])
        if vol > 0.25:
            return None

    size = kelly_size(total_edge, kelly_price, state.cash)
    if size < 0.50 or state.cash < size:
        return None

    # Compute limit price (inside the spread for maker edge)
    if direction == "BUY":
        limit_price = market.best_bid + market.spread * 0.3  # 30% inside spread
    else:
        limit_price = market.best_ask - market.spread * 0.3

    return {
        "market": market,
        "direction": direction,
        "edge": round(total_edge, 4),
        "size": round(size, 2),
        "limit_price": round(limit_price, 4),
        "markov_prob": est.calibrated_probability,
        "confidence": est.confidence,
        "category_mult": cat_mult,
        "smart_money_confirms": sm_confirms,
    }


def check_shadow_fill(position: PaperPosition, market: Market) -> bool:
    """Check if a shadow order would have been filled.

    A maker limit order fills when someone crosses our price level.
    For BUY: fills when ask drops to or below our limit
    For SELL: fills when bid rises to or above our limit
    """
    if position.filled:
        return True
    if position.direction == "BUY":
        return market.best_ask <= position.limit_price
    else:
        return market.best_bid >= position.limit_price


# ─── Telegram Handlers ────────────────────────────────────────────

def build_app(state: BotState, http_session_holder: list) -> Application:
    """Build the Telegram application with all handlers."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    app = Application.builder().token(token).build()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def authorized(update: Update) -> bool:
        if not chat_id:
            return True
        return str(update.effective_chat.id) == chat_id

    # ── /start ──
    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        text = (
            "<b>Polymarket Trading Bot</b>\n\n"
            f"Mode: <b>{state.mode.upper()}</b>\n"
            f"Capital: <b>${state.starting_capital:.2f}</b>\n"
            f"Model: Markov Chain + Bias Calibration\n"
            f"Kelly: Quarter (0.25)\n\n"
            "Use /help for all commands"
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Status", callback_data="cmd:status"),
             InlineKeyboardButton("Scan Now", callback_data="cmd:scan")],
            [InlineKeyboardButton("Positions", callback_data="cmd:positions"),
             InlineKeyboardButton("P&L", callback_data="cmd:pnl")],
        ])
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb)

    # ── /status ──
    async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        dd = 0.0
        eq = [state.starting_capital]
        if state.trades:
            for t in state.trades:
                eq.append(eq[-1] + t.pnl)
            peak = max(eq)
            dd = (peak - eq[-1]) / peak * 100 if peak > 0 else 0

        # Mini equity sparkline
        equity_spark = ""
        if len(eq) > 1:
            blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
            mn, mx = min(eq[-20:]), max(eq[-20:])
            rng = mx - mn
            for v in eq[-20:]:
                idx = int((v - mn) / rng * (len(blocks) - 1)) if rng > 0 else 3
                equity_spark += blocks[idx]
            equity_spark = f"<code>{equity_spark}</code>"

        # Position distribution bar chart
        pos_chart = ""
        if state.positions:
            cats: dict[str, float] = {}
            for p in state.positions.values():
                cats[p.category] = cats.get(p.category, 0) + p.size_usd
            total_pos_val = sum(cats.values()) or 1
            lines = []
            for cat, val in sorted(cats.items(), key=lambda x: -x[1]):
                pct = val / total_pos_val * 100
                filled = int(pct / 100 * 12)
                bar = "\u2588" * filled + "\u2591" * (12 - filled)
                lines.append(f"  {cat:<8} {bar} ${val:.2f}")
            pos_chart = "\n<b>Position Distribution:</b>\n" + "\n".join(lines)

        text = (
            "<b>Portfolio Status</b>\n\n"
            f"Mode:       <b>{state.mode.upper()}</b>\n"
            f"Cash:       ${state.cash:.2f}\n"
            f"Positions:  {len(state.positions)}\n"
            f"Total Value: <b>${state.total_value:.2f}</b>\n"
            f"Total P&L:  <b>${state.total_pnl:+.2f}</b>\n"
            f"Return:     {(state.total_value / state.starting_capital - 1) * 100:+.1f}%\n"
            f"Trades:     {state.n_trades}\n"
            f"Win Rate:   {state.win_rate * 100:.1f}%\n"
            f"Drawdown:   {dd:.1f}%\n"
            f"Markets:    {len(state.markets)}\n"
            f"Scans:      {state.scan_count}\n"
            f"Uptime:     {state.uptime}\n"
            f"Kill Switch: {'ON' if state.kill_switch else 'OFF'}"
        )
        if equity_spark:
            text += f"\n\n<b>Equity Curve:</b>\n{equity_spark}"
        if pos_chart:
            text += f"\n{pos_chart}"

        await update.message.reply_text(text, parse_mode="HTML")

    # ── /positions ──
    async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.positions:
            await update.message.reply_text("No open positions.")
            return
        lines = ["<b>Open Positions</b>\n"]
        for tid, pos in state.positions.items():
            fill_status = "FILLED" if pos.filled else "PENDING"
            m = state.markets.get(pos.market_id)
            current = m.mid_price if m else pos.entry_price
            lines.append(
                f"{'BUY' if pos.direction == 'BUY' else 'SELL NO'} "
                f"<b>{pos.question[:40]}</b>\n"
                f"  Entry: {pos.entry_price:.3f} | Now: {current:.3f} | "
                f"${pos.size_usd:.2f} | {fill_status}\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ── /trades ──
    async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.trades:
            await update.message.reply_text("No trades yet.")
            return
        recent = state.trades[-10:]
        lines = ["<b>Recent Trades</b>\n"]
        for t in reversed(recent):
            emoji = "+" if t.pnl > 0 else "-" if t.pnl < 0 else "~"
            lines.append(
                f"[{emoji}] {t.direction} {t.question[:35]}\n"
                f"  ${t.entry_price:.3f} -> ${t.exit_price:.3f} | "
                f"PnL: <b>${t.pnl:+.2f}</b> | {t.mode}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ── /pnl ──
    async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.trades:
            await update.message.reply_text("No trades to analyze.")
            return

        wins = [t for t in state.trades if t.pnl > 0]
        losses = [t for t in state.trades if t.pnl <= 0]
        buys = [t for t in state.trades if t.direction == "BUY"]
        sells = [t for t in state.trades if t.direction == "SELL"]

        buy_wr = len([t for t in buys if t.pnl > 0]) / max(len(buys), 1) * 100
        sell_wr = len([t for t in sells if t.pnl > 0]) / max(len(sells), 1) * 100

        text = (
            "<b>P&L Breakdown</b>\n\n"
            f"Total P&L:     <b>${state.total_pnl:+.2f}</b>\n"
            f"Return:        {(state.total_value / state.starting_capital - 1) * 100:+.1f}%\n"
            f"Trades:        {len(state.trades)}\n"
            f"Wins/Losses:   {len(wins)}/{len(losses)}\n"
            f"Win Rate:      {state.win_rate * 100:.1f}%\n\n"
            f"<b>By Direction:</b>\n"
            f"  BUY:  {len(buys)} trades, {buy_wr:.0f}% WR\n"
            f"  SELL: {len(sells)} trades, {sell_wr:.0f}% WR\n\n"
            f"<b>By Category:</b>"
        )
        cats = {}
        for t in state.trades:
            if t.category not in cats:
                cats[t.category] = {"n": 0, "pnl": 0.0, "w": 0}
            cats[t.category]["n"] += 1
            cats[t.category]["pnl"] += t.pnl
            if t.pnl > 0:
                cats[t.category]["w"] += 1

        for cat, s in sorted(cats.items(), key=lambda x: -x[1]["pnl"]):
            wr = s["w"] / s["n"] * 100
            text += f"\n  {cat}: {s['n']} trades, {wr:.0f}% WR, ${s['pnl']:+.2f}"

        await update.message.reply_text(text, parse_mode="HTML")

    # ── /scan ──
    async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        await update.message.reply_text("Scanning markets...")
        session = http_session_holder[0]
        if session:
            signals = await run_scan(session, state)
            if signals:
                lines = [f"<b>Found {len(signals)} opportunities:</b>\n"]
                for sig in signals[:5]:
                    m = sig["market"]
                    lines.append(
                        f"{sig['direction']} <b>{m.question[:40]}</b>\n"
                        f"  Edge: {sig['edge']*100:.1f}% | Size: ${sig['size']:.2f} | "
                        f"Conf: {sig['confidence']:.0%}"
                    )
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            else:
                await update.message.reply_text("No opportunities found this scan.")

    # ── /opportunities ──
    async def cmd_opportunities(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.signals:
            await update.message.reply_text("No recent signals. Use /scan first.")
            return
        lines = ["<b>Recent Signals (ranked by edge)</b>\n"]
        for sig in sorted(state.signals, key=lambda s: -s["edge"])[:10]:
            m = sig["market"]
            lines.append(
                f"{sig['direction']} <b>{m.question[:40]}</b>\n"
                f"  Edge: {sig['edge']*100:.1f}% | ${sig['size']:.2f} | "
                f"{m.category} | {sig['confidence']:.0%}\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ── /markets ──
    async def cmd_markets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        cats = {}
        for m in state.markets.values():
            cats[m.category] = cats.get(m.category, 0) + 1
        text = f"<b>Market Universe: {len(state.markets)} markets</b>\n\n"
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            text += f"  {cat}: {n}\n"
        top = sorted(state.markets.values(), key=lambda m: -m.volume_24h)[:5]
        text += "\n<b>Top by volume:</b>\n"
        for m in top:
            text += f"  ${m.volume_24h/1000:.0f}K — {m.question[:45]}\n"
        await update.message.reply_text(text, parse_mode="HTML")

    # ── /mode ──
    async def cmd_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("PAPER", callback_data="mode:paper"),
             InlineKeyboardButton("LIVE", callback_data="mode:live")],
        ])
        await update.message.reply_text(
            f"Current mode: <b>{state.mode.upper()}</b>\nSelect new mode:",
            parse_mode="HTML", reply_markup=kb
        )

    # ── /capital ──
    async def cmd_capital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if state.mode != "paper":
            await update.message.reply_text("Capital can only be set in paper mode.")
            return
        args = ctx.args
        if args and args[0].replace(".", "").isdigit():
            new_cap = float(args[0])
            state.starting_capital = new_cap
            state.cash = new_cap
            state.positions.clear()
            state.trades.clear()
            await update.message.reply_text(
                f"Capital reset to <b>${new_cap:.2f}</b>. Portfolio cleared.",
                parse_mode="HTML"
            )
        else:
            await update.message.reply_text(
                f"Current capital: ${state.starting_capital:.2f}\n"
                "Usage: /capital 100"
            )

    # ── /kill ──
    async def cmd_kill(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        state.kill_switch = True
        await update.message.reply_text(
            "<b>KILL SWITCH ACTIVATED</b>\nAll trading halted. Use /resume to restart.",
            parse_mode="HTML"
        )

    # ── /resume ──
    async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        state.kill_switch = False
        await update.message.reply_text("Trading resumed.")

    # ── /stats ──
    async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        fl = state.combiner.get_fundamental_law_stats()
        text = (
            "<b>Alpha Combination — Fundamental Law</b>\n\n"
            f"IR = IC x sqrt(N)\n\n"
            f"Avg IC:          {fl['avg_ic']:.4f}\n"
            f"Effective N:     {fl['effective_n']:.2f}\n"
            f"Information Ratio: <b>{fl['information_ratio']:.4f}</b>\n"
            f"Strategies:      {fl['n_strategies_tracked']}\n"
            f"With IC data:    {fl['n_strategies_with_ic']}\n"
        )
        if fl["per_strategy_ic"]:
            text += "\n<b>Per-Strategy IC:</b>\n"
            for name, ic in fl["per_strategy_ic"].items():
                text += f"  {name}: {ic:+.4f}\n"
        if fl["optimal_weights"]:
            text += "\n<b>Optimal Weights:</b>\n"
            for name, w in fl["optimal_weights"].items():
                text += f"  {name}: {w:.4f}\n"
        await update.message.reply_text(text, parse_mode="HTML")

    # ── /arb ── Combinatorial arbitrage scan
    async def cmd_arb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        await update.message.reply_text("Scanning for combinatorial arbitrage...")

        try:
            from core.combinatorial_arb import CombinatorialArbEngine

            engine = CombinatorialArbEngine(min_profit_pct=0.3)
            market_dicts = []
            for m in state.markets.values():
                market_dicts.append({
                    "condition_id": m.condition_id,
                    "question": m.question,
                    "slug": m.slug,
                    "category": m.category,
                    "outcomes": m.outcomes,
                    "outcome_prices": [m.yes_price, m.no_price],
                    "tokens": [{"token_id": t} for t in m.token_ids],
                    "tags": [],
                    "liquidity": m.liquidity,
                    "volume_24h": m.volume_24h,
                })

            clusters = engine.build_clusters(market_dicts)
            opps = engine.detect_arbitrage(clusters)
            stats = engine.get_stats()

            text = f"<b>Combinatorial Arbitrage Scan</b>\n\n"
            text += f"Markets scanned: {len(state.markets)}\n"
            text += f"Clusters found: {stats['total_clusters']}\n"
            text += f"  Mutex: {stats['mutex_clusters']}\n"
            text += f"  Rebalancing: {stats['rebalancing_clusters']}\n"
            text += f"Opportunities: {len(opps)}\n"

            if opps:
                text += f"\n<b>Top Opportunities:</b>\n"
                for opp in opps[:5]:
                    text += (
                        f"\n{opp.arb_type} | ROI: {opp.roi_pct:+.2f}%\n"
                        f"  Profit: ${opp.guaranteed_profit:.4f}\n"
                        f"  Cost: ${opp.total_cost:.4f}\n"
                        f"  Legs: {len(opp.legs)}\n"
                        f"  {opp.description[:60]}\n"
                    )
            else:
                text += "\nNo profitable arbitrage found at current prices."

            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Arb scan error: {e}")

    # ── /risk ── Show risk rules and status
    async def cmd_risk(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        n_pos = len(state.positions)
        total_value = state.total_value
        peak = state.peak_value
        dd = (peak - total_value) / peak * 100 if peak > 0 else 0
        cash_ratio = state.cash / max(state.starting_capital, 0.01) * 100
        daily_loss = 0.0
        if state.daily_start_value > 0:
            daily_loss = (state.daily_start_value - total_value) / state.daily_start_value * 100

        text = (
            f"<b>Risk Status</b>\n\n"
            f"Positions: {n_pos}/{MAX_CONCURRENT_POSITIONS} max\n"
            f"Drawdown: {dd:.1f}% (limit: {MAX_DRAWDOWN_PCT*100:.0f}%)\n"
            f"Daily loss: {daily_loss:.1f}% (limit: {DAILY_LOSS_LIMIT_PCT*100:.0f}%)\n"
            f"Cash ratio: {cash_ratio:.0f}%\n"
            f"Peak value: ${peak:.2f}\n\n"
            f"<b>Risk Rules (enforced):</b>\n"
            f"  Max position: {MAX_POSITION_PCT*100:.0f}% of portfolio\n"
            f"  Max drawdown: {MAX_DRAWDOWN_PCT*100:.0f}%\n"
            f"  Max concurrent: {MAX_CONCURRENT_POSITIONS} positions\n"
            f"  Daily loss limit: {DAILY_LOSS_LIMIT_PCT*100:.0f}%\n"
            f"  Low-cash mode: edge > {HIGH_CONVICTION_EDGE*100:.0f}% only "
            f"(when cash < {HIGH_CONVICTION_CASH_PCT*100:.0f}%)\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ── /backtest ──
    async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        n_markets = 50
        if ctx.args and ctx.args[0].isdigit():
            n_markets = int(ctx.args[0])
        n_markets = max(5, min(n_markets, 500))

        await update.message.reply_text(
            f"Running mini-backtest on last {n_markets} resolved markets..."
        )

        async def _run_backtest():
            try:
                rows = state.db.execute(
                    "SELECT entry_price, exit_price, size_usd, edge, pnl "
                    "FROM trades ORDER BY id DESC LIMIT ?",
                    (n_markets,),
                ).fetchall()

                if not rows:
                    return "No historical trades found for backtest."

                pnls = [r[4] for r in rows]
                sizes = [r[2] for r in rows]
                edges = [r[3] for r in rows]
                wins = sum(1 for p in pnls if p > 0)
                total_pnl = sum(pnls)
                wr = wins / len(pnls) * 100

                # Sharpe ratio (annualized, assuming ~3 trades/day)
                if len(pnls) > 1:
                    returns = [p / max(s, 0.01) for p, s in zip(pnls, sizes)]
                    avg_r = np.mean(returns)
                    std_r = np.std(returns)
                    sharpe = (avg_r / std_r * np.sqrt(365 * 3)) if std_r > 0 else 0.0
                else:
                    sharpe = 0.0

                # Max drawdown
                equity = [state.starting_capital]
                for p in reversed(pnls):
                    equity.append(equity[-1] + p)
                peak = equity[0]
                max_dd = 0.0
                for e in equity:
                    peak = max(peak, e)
                    dd = (peak - e) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)

                avg_edge = np.mean(edges) * 100
                avg_size = np.mean(sizes)

                # Equity sparkline
                sparkline = _text_sparkline(equity[-20:]) if len(equity) > 1 else ""

                text = (
                    f"<b>Mini-Backtest Results</b>\n"
                    f"Trades: {len(pnls)}\n\n"
                    f"{sparkline}\n\n"
                    f"Win Rate:    <b>{wr:.1f}%</b>\n"
                    f"Total P&L:   <b>${total_pnl:+.2f}</b>\n"
                    f"Sharpe:      <b>{sharpe:.2f}</b>\n"
                    f"Max DD:      {max_dd:.1f}%\n"
                    f"Avg Edge:    {avg_edge:.2f}%\n"
                    f"Avg Size:    ${avg_size:.2f}\n"
                )
                return text
            except Exception as e:
                return f"Backtest error: {e}"

        result = await _run_backtest()
        await update.message.reply_text(result, parse_mode="HTML")

    # ── /alerts ──
    async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        args = ctx.args

        if not args:
            status = "ON" if state.alerts_enabled else "OFF"
            text = (
                f"<b>Alert Settings</b>\n\n"
                f"Status:    <b>{status}</b>\n"
                f"Threshold: <b>{state.alerts_threshold:.1f}%</b>\n"
                f"Tracked:   {len(state.price_snapshots)} markets\n\n"
                f"Usage:\n"
                f"  /alerts on — Enable alerts\n"
                f"  /alerts off — Disable alerts\n"
                f"  /alerts threshold 5 — Set % threshold"
            )
            await update.message.reply_text(text, parse_mode="HTML")
            return

        subcmd = args[0].lower()
        if subcmd == "on":
            state.alerts_enabled = True
            # Snapshot current prices
            for cid, m in state.markets.items():
                if m.mid_price > 0:
                    state.price_snapshots[cid] = m.mid_price
            await update.message.reply_text(
                f"Alerts <b>enabled</b>. Tracking {len(state.price_snapshots)} markets "
                f"with {state.alerts_threshold:.1f}% threshold.",
                parse_mode="HTML",
            )
        elif subcmd == "off":
            state.alerts_enabled = False
            await update.message.reply_text("Alerts <b>disabled</b>.", parse_mode="HTML")
        elif subcmd == "threshold" and len(args) > 1:
            try:
                val = float(args[1])
                state.alerts_threshold = max(0.5, min(val, 50.0))
                await update.message.reply_text(
                    f"Alert threshold set to <b>{state.alerts_threshold:.1f}%</b>.",
                    parse_mode="HTML",
                )
            except ValueError:
                await update.message.reply_text("Usage: /alerts threshold 5")
        else:
            await update.message.reply_text(
                "Usage: /alerts [on|off|threshold <pct>]"
            )

    # ── /performance ──
    async def cmd_performance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.trades:
            await update.message.reply_text("No trades yet for performance analysis.")
            return

        # Equity curve
        equity = [state.starting_capital]
        for t in state.trades:
            equity.append(equity[-1] + t.pnl)
        sparkline = _text_sparkline(equity[-30:])

        # Win rate by category
        cats: dict[str, dict] = {}
        for t in state.trades:
            if t.category not in cats:
                cats[t.category] = {"n": 0, "w": 0, "pnl": 0.0}
            cats[t.category]["n"] += 1
            cats[t.category]["pnl"] += t.pnl
            if t.pnl > 0:
                cats[t.category]["w"] += 1

        cat_lines = []
        for cat, s in sorted(cats.items(), key=lambda x: -x[1]["pnl"]):
            wr = s["w"] / s["n"] * 100 if s["n"] > 0 else 0
            bar = _bar_char(wr, 100, width=8)
            cat_lines.append(f"  {cat:<10} {bar} {wr:.0f}% ({s['n']}t, ${s['pnl']:+.2f})")

        # Best / worst trades
        sorted_by_pnl = sorted(state.trades, key=lambda t: t.pnl)
        worst = sorted_by_pnl[:3]
        best = sorted_by_pnl[-3:][::-1]

        best_lines = []
        for t in best:
            best_lines.append(f"  ${t.pnl:+.2f} — {t.question[:35]}")
        worst_lines = []
        for t in worst:
            worst_lines.append(f"  ${t.pnl:+.2f} — {t.question[:35]}")

        # Rolling 7-day Sharpe
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        recent = []
        for t in state.trades:
            try:
                ts = datetime.fromisoformat(t.timestamp)
                if ts >= week_ago:
                    ret = t.pnl / max(t.size_usd, 0.01)
                    recent.append(ret)
            except Exception:
                pass

        if len(recent) > 1:
            avg_r = np.mean(recent)
            std_r = np.std(recent)
            sharpe_7d = (avg_r / std_r * np.sqrt(365 * 3)) if std_r > 0 else 0.0
        elif len(recent) == 1:
            sharpe_7d = 0.0
        else:
            sharpe_7d = 0.0

        text = (
            f"<b>Performance Analytics</b>\n\n"
            f"<b>Equity Curve (last 30):</b>\n"
            f"<code>{sparkline}</code>\n\n"
            f"<b>Win Rate by Category:</b>\n"
            + "\n".join(cat_lines) + "\n\n"
            f"<b>Best Trades:</b>\n"
            + "\n".join(best_lines) + "\n\n"
            f"<b>Worst Trades:</b>\n"
            + "\n".join(worst_lines) + "\n\n"
            f"<b>7-Day Rolling Sharpe:</b> {sharpe_7d:.2f}\n"
            f"<b>7-Day Trades:</b> {len(recent)}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ── /export ──
    async def cmd_export(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        if not state.trades:
            await update.message.reply_text("No trades to export.")
            return

        fmt = "csv"
        if ctx.args and ctx.args[0].lower() in ("csv", "json"):
            fmt = ctx.args[0].lower()

        trades_data = []
        for t in state.trades:
            trades_data.append({
                "timestamp": t.timestamp,
                "market_id": t.market_id,
                "question": t.question,
                "category": t.category,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size_usd": t.size_usd,
                "edge": t.edge,
                "pnl": t.pnl,
                "mode": t.mode,
            })

        if fmt == "json":
            content = json.dumps(trades_data, indent=2)
            filename = "trades_export.json"
        else:
            buf = io.StringIO()
            if trades_data:
                writer = csv.DictWriter(buf, fieldnames=trades_data[0].keys())
                writer.writeheader()
                writer.writerows(trades_data)
            content = buf.getvalue()
            filename = "trades_export.csv"

        # Send as file attachment
        file_bytes = io.BytesIO(content.encode("utf-8"))
        file_bytes.name = filename
        await update.message.reply_document(
            document=file_bytes,
            filename=filename,
            caption=f"Exported {len(trades_data)} trades as {fmt.upper()}.",
        )

    # ── /weights ── Bayesian strategy weight posteriors
    async def cmd_weights(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        summary = state.bayesian_updater.get_strategy_summary()
        if not summary:
            await update.message.reply_text("No strategy data yet. Trades must resolve first.")
            return

        weights = state.bayesian_updater.get_optimal_weights()
        lines = ["<b>Bayesian Strategy Weights</b>\n"]

        for name, info in summary.items():
            ci_lo, ci_hi = info["credible_interval_95"]
            regime = " [REGIME]" if info["regime_change"] else ""
            lines.append(
                f"<b>{name}</b>{regime}\n"
                f"  Weight: {weights.get(name, 0):.4f}\n"
                f"  WR: {info['posterior_win_rate']:.3f} "
                f"[{ci_lo:.3f}, {ci_hi:.3f}]\n"
                f"  IC: {info['rolling_ic']:+.4f} | "
                f"Edge: {info['expected_edge']:+.4f}\n"
                f"  Beta({info['beta_alpha']:.1f}, {info['beta_beta']:.1f}) | "
                f"N={info['n_observations']}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ── /smartmoney ── Recent large order flow
    async def cmd_smartmoney(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        flows = state.smart_money.get_all_flows()
        if not flows:
            await update.message.reply_text(
                "No smart money flow detected yet. Need more orderbook data."
            )
            return

        lines = ["<b>Smart Money Flow</b>\n"]
        # Sort by absolute net flow
        sorted_flows = sorted(
            flows.items(), key=lambda x: abs(x[1].net_flow_usd), reverse=True
        )

        for market_id, flow in sorted_flows[:10]:
            m = state.markets.get(market_id)
            question = m.question[:40] if m else market_id[:20]
            direction = "BUY" if flow.flow_imbalance > 0 else "SELL"
            lines.append(
                f"<b>{question}</b>\n"
                f"  {direction} pressure | Imbalance: {flow.flow_imbalance:+.3f}\n"
                f"  Net: ${flow.net_flow_usd:+,.0f} | "
                f"Buys: {flow.n_large_buys} (${flow.buy_volume_usd:,.0f}) | "
                f"Sells: {flow.n_large_sells} (${flow.sell_volume_usd:,.0f})\n"
                f"  VWAP B/S: {flow.vwap_buy:.3f}/{flow.vwap_sell:.3f} | "
                f"Avg: ${flow.avg_order_size:,.0f}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ── Helper: Text sparkline ──
    def _text_sparkline(values: list[float]) -> str:
        """Generate a Unicode sparkline from a list of values."""
        if not values or len(values) < 2:
            return ""
        blocks = " _.,:-=!#"
        mn, mx = min(values), max(values)
        rng = mx - mn
        if rng == 0:
            return blocks[4] * len(values)
        result = ""
        for v in values:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            result += blocks[idx]
        return result

    # ── Helper: Bar character ──
    def _bar_char(value: float, max_val: float, width: int = 10) -> str:
        """Generate a Unicode bar chart segment."""
        filled = int(value / max(max_val, 0.01) * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    # ── /help ──
    async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not authorized(update):
            return
        text = (
            "<b>Command Reference</b>\n\n"
            "/start — Welcome\n"
            "/status — Portfolio overview\n"
            "/positions — Open positions\n"
            "/trades — Recent trades\n"
            "/pnl — P&L breakdown\n"
            "/scan — Manual market scan\n"
            "/opportunities — Current signals\n"
            "/arb — Combinatorial arbitrage scan\n"
            "/markets — Market universe\n"
            "/mode — Toggle paper/live\n"
            "/capital &lt;amt&gt; — Set paper capital\n"
            "/kill — Emergency stop\n"
            "/resume — Resume trading\n"
            "/stats — Alpha combiner stats\n"
            "/risk — Risk rules &amp; status\n"
            "/backtest &lt;n&gt; — Mini-backtest on last n trades\n"
            "/alerts — Configure price alerts\n"
            "/performance — Rich analytics &amp; sparklines\n"
            "/export &lt;csv|json&gt; — Export trade history\n"
            "/weights — Bayesian strategy weight posteriors\n"
            "/smartmoney — Recent large order flow\n"
            "/help — This message"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ── Callbacks ──
    async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data or ""

        if data.startswith("mode:"):
            new_mode = data.split(":")[1]
            if new_mode == "live" and not os.getenv("POLYMARKET_PRIVATE_KEY"):
                await query.edit_message_text(
                    "Cannot switch to LIVE: POLYMARKET_PRIVATE_KEY not set.",
                    parse_mode="HTML"
                )
                return
            state.mode = new_mode
            await query.edit_message_text(
                f"Mode switched to <b>{new_mode.upper()}</b>",
                parse_mode="HTML"
            )

    # Register all handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("opportunities", cmd_opportunities))
    app.add_handler(CommandHandler("markets", cmd_markets))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("capital", cmd_capital))
    app.add_handler(CommandHandler("kill", cmd_kill))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("arb", cmd_arb))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("weights", cmd_weights))
    app.add_handler(CommandHandler("smartmoney", cmd_smartmoney))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(handle_callback))

    return app


# ─── Scan & Trade Loop ────────────────────────────────────────────

async def run_scan(session: aiohttp.ClientSession, state: BotState) -> list[dict]:
    """Run a full market scan cycle."""
    state.scan_count += 1
    signals = []

    # 1. Fetch market universe
    raw_markets = await fetch_markets(session)
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

        prices = raw.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = [float(p) for p in json.loads(prices)]
            except:
                prices = []
        elif isinstance(prices, list):
            prices = [float(p) for p in prices]

        tags = raw.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = []

        question = raw.get("question", "")[:100]
        category = classify(f"{question} {raw.get('description', '')}", [str(t) for t in (tags or [])])

        market = Market(
            condition_id=cid, question=question, slug=raw.get("slug", ""),
            category=category, outcomes=outcomes,
            token_ids=[str(t) for t in token_ids],
            liquidity=float(raw.get("liquidity", 0) or 0),
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            end_date=raw.get("endDate", ""),
            yes_price=prices[0] if prices else 0,
            no_price=prices[1] if len(prices) > 1 else 0,
            mid_price=prices[0] if prices else 0,
            # Structural fields for rigorous combinatorial arb detection
            event_slug=raw.get("eventSlug", "") or "",
            neg_risk_market_id=raw.get("negRiskMarketID", "") or "",
            group_item_title=raw.get("groupItemTitle", "") or "",
        )

        # Preserve history from previous scans
        if cid in state.markets and state.markets[cid].price_history:
            market.price_history = state.markets[cid].price_history
        state.markets[cid] = market

    # 2. Fetch price history for markets missing it
    for m in list(state.markets.values()):
        if len(m.price_history) < 15 and m.token_ids:
            history = await fetch_history(session, m.token_ids[0])
            if history:
                m.price_history = history
            await asyncio.sleep(0.1)

    # 2b. Ensure WSFeed is tracking tokens for top markets
    if state.ws_feed:
        for m in sorted(state.markets.values(), key=lambda x: -x.volume_24h)[:30]:
            if m.token_ids:
                outcome = m.outcomes[0] if m.outcomes else "Yes"
                state.ws_feed.track_token(m.token_ids[0], m.condition_id, outcome)

    # 3. Fetch live orderbooks for top markets
    #    If WSFeed is active, use its snapshots instead of REST polling
    top = sorted(state.markets.values(), key=lambda m: -m.volume_24h)[:30]
    for m in top:
        if not m.token_ids:
            continue

        ws_snap = None
        if state.ws_feed and state.ws_feed.is_ws_connected:
            ws_snap = state.ws_feed.get_snapshot(m.token_ids[0])

        if ws_snap:
            # Use WebSocket snapshot — no REST call needed
            m.best_bid = ws_snap.best_bid
            m.best_ask = ws_snap.best_ask
            m.mid_price = ws_snap.mid_price if ws_snap.mid_price > 0 else m.yes_price
            m.spread = ws_snap.spread
            if m.mid_price > 0:
                m.price_history.append(m.mid_price)
            # Ingest bid/ask levels for smart money detection
            for price, size in ws_snap.bid_levels:
                state.smart_money.ingest_trade(
                    m.condition_id, m.token_ids[0], "BUY", price * size, price,
                )
            for price, size in ws_snap.ask_levels:
                state.smart_money.ingest_trade(
                    m.condition_id, m.token_ids[0], "SELL", price * size, price,
                )
        else:
            book = await fetch_book(session, m.token_ids[0])
            if book:
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                m.best_bid = float(bids[0]["price"]) if bids else 0
                m.best_ask = float(asks[0]["price"]) if asks else 1
                m.mid_price = (m.best_bid + m.best_ask) / 2 if (m.best_bid + m.best_ask) > 0 else m.yes_price
                m.spread = m.best_ask - m.best_bid
                if m.mid_price > 0:
                    m.price_history.append(m.mid_price)
                # Ingest orderbook levels for smart money detection
                for b in bids:
                    price_val = float(b.get("price", 0))
                    size_val = float(b.get("size", 0))
                    state.smart_money.ingest_trade(
                        m.condition_id, m.token_ids[0], "BUY",
                        price_val * size_val, price_val,
                    )
                for a in asks:
                    price_val = float(a.get("price", 0))
                    size_val = float(a.get("size", 0))
                    state.smart_money.ingest_trade(
                        m.condition_id, m.token_ids[0], "SELL",
                        price_val * size_val, price_val,
                    )
            await asyncio.sleep(0.1)

    # 4. Check shadow fills + expire stale unfilled orders
    ORDER_TIMEOUT_S = 600  # Cancel unfilled limit orders after 10 minutes
    now = datetime.now(timezone.utc)

    for tid, pos in list(state.positions.items()):
        m = state.markets.get(pos.market_id)

        # Expire unfilled orders that are too old
        if not pos.filled and pos.opened_at:
            try:
                opened = datetime.fromisoformat(pos.opened_at)
                age_s = (now - opened).total_seconds()
                if age_s > ORDER_TIMEOUT_S:
                    logger.info(
                        f"ORDER EXPIRED ({age_s:.0f}s): {pos.direction} {pos.question[:40]}"
                    )
                    state.cash += pos.size_usd  # Return reserved capital
                    del state.positions[tid]
                    continue
            except (ValueError, TypeError):
                pass

        # Check shadow fill (did price cross our limit?)
        if m and not pos.filled and check_shadow_fill(pos, m):
            pos.filled = True
            pos.fill_time = now.isoformat()
            logger.info(f"Shadow fill: {pos.direction} {pos.question[:40]}")

        # Staleness check: if book hasn't updated, don't trust the fill
        if m and pos.filled and m.spread > 0.10:
            logger.warning(
                f"Wide spread ({m.spread:.2f}) on filled position {pos.question[:30]} "
                f"- book may be stale"
            )

        # Check for resolution (price at 0 or 1)
        if m and (m.mid_price >= 0.98 or m.mid_price <= 0.02):
            resolution = 1.0 if m.mid_price >= 0.98 else 0.0
            if pos.direction == "BUY":
                shares = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0
                pnl = shares * resolution - pos.size_usd
            else:
                no_price = 1.0 - pos.entry_price
                shares = pos.size_usd / no_price if no_price > 0 else 0
                pnl = shares * (1.0 - resolution) - pos.size_usd

            state.cash += pos.size_usd + pnl
            trade = TradeRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                market_id=pos.market_id, question=pos.question,
                category=pos.category, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=resolution,
                size_usd=pos.size_usd, edge=pos.edge_at_entry,
                pnl=round(pnl, 2), mode=state.mode.upper(),
            )
            state.log_trade(trade)

            # --- Bayesian updater: learn from resolved trade ---
            won = pnl > 0
            actual_return = pnl / max(pos.size_usd, 0.01)
            # Update all contributing strategies (attribute to primary)
            state.bayesian_updater.update("markov_mc", won, actual_return)
            if pos.direction == "SELL":
                state.bayesian_updater.update("no_bias", won, actual_return)
            state.bayesian_updater.update("maker_edge", won, actual_return)
            # Push updated weights into the combiner
            state.bayesian_updater.apply_to_combiner(state.combiner)

            # Clean up smart money data for resolved market
            state.smart_money.clear_market(pos.market_id)

            del state.positions[tid]

    # 5. Generate new signals (with smart money confirmation)
    if not state.kill_switch:
        for m in top:
            sm_sig = state.smart_money.get_signal(m.condition_id, m.mid_price)
            sig = evaluate_market(m, state, smart_money_signal=sm_sig)
            if sig:
                signals.append(sig)

        # Update peak value for risk tracking
        state.peak_value = max(state.peak_value, state.total_value)

        # Execute top signals (max 3 per scan) with risk checks + book walking
        for sig in sorted(signals, key=lambda s: -s["edge"])[:3]:
            m = sig["market"]
            token_id = m.token_ids[0] if m.token_ids else m.condition_id

            # Risk check (ported from live_paper_trader.py)
            allowed, reason = check_risk(
                size_usd=sig["size"],
                edge=sig["edge"],
                cash=state.cash,
                total_value=state.total_value,
                starting_capital=state.starting_capital,
                peak_value=state.peak_value,
                n_positions=len(state.positions),
                daily_start_value=state.daily_start_value,
            )
            if not allowed:
                logger.info(f"RISK BLOCKED: {reason} | {m.question[:40]}")
                continue

            # Walk the order book for realistic VWAP fill price
            entry_price = m.mid_price
            actual_size = sig["size"]
            book = await fetch_book(session, token_id)
            if book:
                if sig["direction"] == "BUY":
                    asks = book.get("asks", [])
                    fill_price, filled_usd = walk_book(asks, sig["size"], "buy")
                else:
                    bids = book.get("bids", [])
                    fill_price, filled_usd = walk_book(bids, sig["size"], "sell")

                if fill_price > 0 and filled_usd > 0:
                    entry_price = fill_price
                    actual_size = min(sig["size"], filled_usd)
                else:
                    logger.info(f"NO FILL: empty book for {m.question[:40]}")
                    continue

            pos = PaperPosition(
                token_id=token_id,
                market_id=m.condition_id,
                question=m.question,
                category=m.category,
                direction=sig["direction"],
                entry_price=entry_price,
                size_usd=actual_size,
                edge_at_entry=sig["edge"],
                limit_price=sig["limit_price"],
                filled=False,
                opened_at=datetime.now(timezone.utc).isoformat(),
            )

            state.positions[token_id] = pos
            state.cash -= actual_size

    state.signals = signals
    state.snapshot()
    state.persist_state()
    return signals


# ─── Main ──────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Polymarket Telegram Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--capital", type=float, default=50.0)
    parser.add_argument("--interval", type=int, default=120, help="Scan interval seconds")
    parser.add_argument(
        "--ws", action="store_true", default=False,
        help="Use WebSocket feed for real-time orderbook data instead of REST polling",
    )
    args = parser.parse_args()

    state = BotState(mode=args.mode, capital=args.capital)
    http_session_holder = [None]

    # Initialize WSFeed if --ws flag is set
    if args.ws:
        state.ws_feed = WSFeed()

    logger.info("Starting Polymarket Bot...")
    logger.info(f"  Mode: {args.mode.upper()}")
    logger.info(f"  Capital: ${args.capital:.2f}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  WebSocket: {'ON' if args.ws else 'OFF'}")

    app = build_app(state, http_session_holder)

    # Initialize telegram
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started. Send /start to your bot.")

    # Send startup message
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if chat_id:
        await app.bot.send_message(
            chat_id=chat_id,
            text=(
                f"<b>Bot Started</b>\n\n"
                f"Mode: {args.mode.upper()}\n"
                f"Capital: ${args.capital:.2f}\n"
                f"Interval: {args.interval}s\n\n"
                f"Use /help for commands"
            ),
            parse_mode="HTML",
        )

    # Start WSFeed if enabled
    if state.ws_feed:
        await state.ws_feed.start()
        logger.info("WSFeed started — real-time orderbook via WebSocket")

    # Main scan loop
    async with aiohttp.ClientSession() as session:
        http_session_holder[0] = session
        try:
            while True:
                try:
                    signals = await run_scan(session, state)
                    if signals and chat_id:
                        for sig in signals[:2]:
                            m = sig["market"]
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=(
                                    f"<b>SIGNAL: {sig['direction']}</b>\n"
                                    f"{m.question[:60]}\n"
                                    f"Edge: {sig['edge']*100:.1f}% | "
                                    f"Size: ${sig['size']:.2f} | "
                                    f"Cat: {m.category}"
                                ),
                                parse_mode="HTML",
                            )

                    # Check price movement alerts
                    if state.alerts_enabled and chat_id:
                        alert_lines = []
                        for cid, m in state.markets.items():
                            if m.mid_price <= 0:
                                continue
                            old_price = state.price_snapshots.get(cid)
                            if old_price is not None and old_price > 0:
                                move_pct = abs(m.mid_price - old_price) / old_price * 100
                                if move_pct >= state.alerts_threshold:
                                    direction = "\u2b06" if m.mid_price > old_price else "\u2b07"
                                    alert_lines.append(
                                        f"{direction} <b>{m.question[:45]}</b>\n"
                                        f"  {old_price:.3f} -> {m.mid_price:.3f} ({move_pct:+.1f}%)"
                                    )
                            state.price_snapshots[cid] = m.mid_price

                        if alert_lines:
                            alert_text = (
                                f"<b>Price Alert ({state.alerts_threshold:.1f}% threshold)</b>\n\n"
                                + "\n".join(alert_lines[:10])
                            )
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=alert_text,
                                parse_mode="HTML",
                            )
                except Exception as e:
                    logger.error(f"Scan error: {e}")

                await asyncio.sleep(args.interval)
        except asyncio.CancelledError:
            pass
        finally:
            http_session_holder[0] = None

    # Cleanup
    if state.ws_feed:
        await state.ws_feed.stop()
        logger.info("WSFeed stopped")
    await app.updater.stop()
    await app.stop()
    await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")
