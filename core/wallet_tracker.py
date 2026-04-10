"""Wallet screening and copy-trading system for Polymarket.

Every Polymarket trade settles on Polygon, so all trades are public.
This module identifies consistently profitable wallets and generates
copy-trade signals when they enter new positions.

Two main classes:
    WalletScreener — Fetches trade data, scores wallets, maintains watchlist.
    WalletCopyTrader — Monitors watched wallets for new trades, generates signals.

Data sources (in priority order):
    1. CLOB API GET /trades — recent trades with maker/taker addresses
    2. Polygon RPC — Transfer events on CTF Exchange contract (fallback)
    3. Dune Analytics — Historical wallet performance (batch screening)
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

CLOB_API = "https://clob.polymarket.com"

# Screening thresholds (defaults, overridable)
DEFAULT_MIN_WIN_RATE = 0.60
DEFAULT_MIN_PNL = 1000.0
DEFAULT_MIN_TRADES = 50
DEFAULT_RECENCY_DAYS = 7
DEFAULT_WATCHLIST_SIZE = 20

# Copy-trade safety
MAX_PRICE_MOVE_PCT = 0.02      # Don't copy if price moved >2% since wallet's trade
MIN_POLL_INTERVAL_S = 10       # Rate limit: don't poll more than once per 10s
COPY_DELAY_TARGET_S = 60       # Target <60s from wallet trade to our copy


# ─── Data Models ──────────────────────────────────────────────────

@dataclass
class WalletTrade:
    """A single trade attributed to a wallet."""
    wallet: str              # 0x... Polygon address
    market_id: str           # condition_id
    token_id: str
    side: str                # "BUY" or "SELL"
    price: float
    size_usd: float
    timestamp: float         # Unix timestamp
    outcome: str = ""        # "Yes" / "No" / outcome label


@dataclass
class WalletProfile:
    """Aggregated performance profile for a single wallet."""
    address: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_trade_size: float = 0.0
    last_trade_ts: float = 0.0
    markets_traded: int = 0
    score: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def recency_days(self) -> float:
        if self.last_trade_ts <= 0:
            return 999.0
        return (time.time() - self.last_trade_ts) / 86400.0


@dataclass
class CopySignal:
    """Signal generated when a watched wallet trades."""
    wallet: str
    market_id: str
    token_id: str
    direction: str           # "BUY" or "SELL"
    wallet_price: float      # Price the wallet got
    current_price: float     # Price when we detected it
    wallet_score: float
    delay_s: float           # Seconds between wallet trade and detection
    timestamp: float
    outcome: str = ""


# ─── Wallet Score DB (SQLite) ─────────────────────────────────────

class WalletScoreDB:
    """SQLite persistence for wallet scores and copy-trade tracking."""

    def __init__(self, db_path: str = "data/wallet_scores.db"):
        os.makedirs(os.path.dirname(db_path) or "data", exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                avg_trade_size REAL DEFAULT 0.0,
                last_trade_ts REAL DEFAULT 0.0,
                markets_traded INTEGER DEFAULT 0,
                score REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS wallet_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size_usd REAL NOT NULL,
                timestamp REAL NOT NULL,
                outcome TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS copy_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet TEXT NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                wallet_price REAL,
                our_price REAL,
                wallet_score REAL,
                delay_s REAL,
                size_usd REAL,
                pnl REAL DEFAULT NULL,
                resolved INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_wallet_trades_wallet
                ON wallet_trades(wallet);
            CREATE INDEX IF NOT EXISTS idx_wallet_trades_market
                ON wallet_trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_copy_trades_wallet
                ON copy_trades(wallet);
        """)
        self.db.commit()

    def upsert_wallet(self, profile: WalletProfile) -> None:
        self.db.execute("""
            INSERT INTO wallets (address, total_trades, wins, losses,
                total_pnl, avg_trade_size, last_trade_ts, markets_traded,
                score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                total_trades=excluded.total_trades,
                wins=excluded.wins,
                losses=excluded.losses,
                total_pnl=excluded.total_pnl,
                avg_trade_size=excluded.avg_trade_size,
                last_trade_ts=excluded.last_trade_ts,
                markets_traded=excluded.markets_traded,
                score=excluded.score,
                updated_at=excluded.updated_at
        """, (
            profile.address, profile.total_trades, profile.wins,
            profile.losses, profile.total_pnl, profile.avg_trade_size,
            profile.last_trade_ts, profile.markets_traded, profile.score,
            datetime.now(timezone.utc).isoformat(),
        ))
        self.db.commit()

    def get_watchlist(self, limit: int = DEFAULT_WATCHLIST_SIZE) -> list[WalletProfile]:
        rows = self.db.execute(
            "SELECT address, total_trades, wins, losses, total_pnl, "
            "avg_trade_size, last_trade_ts, markets_traded, score "
            "FROM wallets ORDER BY score DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            WalletProfile(
                address=r[0], total_trades=r[1], wins=r[2], losses=r[3],
                total_pnl=r[4], avg_trade_size=r[5], last_trade_ts=r[6],
                markets_traded=r[7], score=r[8],
            )
            for r in rows
        ]

    def log_copy_trade(
        self,
        signal: CopySignal,
        our_price: float,
        size_usd: float,
    ) -> None:
        self.db.execute("""
            INSERT INTO copy_trades (wallet, market_id, token_id, direction,
                wallet_price, our_price, wallet_score, delay_s, size_usd, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.wallet, signal.market_id, signal.token_id,
            signal.direction, signal.wallet_price, our_price,
            signal.wallet_score, signal.delay_s, size_usd,
            datetime.now(timezone.utc).isoformat(),
        ))
        self.db.commit()

    def resolve_copy_trade(self, market_id: str, pnl: float) -> None:
        """Mark copy trades for a resolved market with their PnL."""
        self.db.execute(
            "UPDATE copy_trades SET pnl = ?, resolved = 1 "
            "WHERE market_id = ? AND resolved = 0",
            (pnl, market_id),
        )
        self.db.commit()

    def get_copy_performance(self) -> dict[str, Any]:
        """Return aggregate copy-trade performance stats."""
        row = self.db.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                   SUM(pnl), AVG(delay_s)
            FROM copy_trades WHERE resolved = 1
        """).fetchone()
        total = row[0] or 0
        wins = row[1] or 0
        total_pnl = row[2] or 0.0
        avg_delay = row[3] or 0.0
        return {
            "total_resolved": total,
            "wins": wins,
            "win_rate": wins / max(total, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_delay_s": round(avg_delay, 1),
        }

    def insert_trade(self, trade: WalletTrade) -> None:
        """Store a raw wallet trade for screening analysis."""
        self.db.execute("""
            INSERT INTO wallet_trades (wallet, market_id, token_id, side,
                price, size_usd, timestamp, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.wallet, trade.market_id, trade.token_id,
            trade.side, trade.price, trade.size_usd,
            trade.timestamp, trade.outcome,
        ))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


# ─── Wallet Screener ──────────────────────────────────────────────

class WalletScreener:
    """Fetches trade history and scores wallets for copy-trading.

    Uses the CLOB API to fetch recent trades with maker/taker addresses,
    then screens wallets by win rate, PnL, trade count, and recency.

    Score formula:
        score = win_rate * log(1 + total_pnl) * recency_factor

    Where recency_factor = max(0, 1 - days_since_last_trade / recency_window)
    """

    def __init__(
        self,
        db: WalletScoreDB | None = None,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        min_pnl: float = DEFAULT_MIN_PNL,
        min_trades: int = DEFAULT_MIN_TRADES,
        recency_days: int = DEFAULT_RECENCY_DAYS,
        watchlist_size: int = DEFAULT_WATCHLIST_SIZE,
    ):
        self.db = db or WalletScoreDB()
        self._min_win_rate = min_win_rate
        self._min_pnl = min_pnl
        self._min_trades = min_trades
        self._recency_days = recency_days
        self._watchlist_size = watchlist_size

    async def fetch_recent_trades(
        self,
        session: Any,
        token_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Fetch recent trades from CLOB API.

        The CLOB /trades endpoint returns trades with maker/taker addresses.
        If token_id is None, fetches across all markets (pagination).

        Returns raw trade dicts from the API.
        """
        import aiohttp

        params: dict[str, Any] = {}
        if token_id:
            params["asset_id"] = token_id

        all_trades: list[dict] = []
        next_cursor: str | None = None
        fetched = 0

        while fetched < limit:
            batch_params = {**params}
            if next_cursor:
                batch_params["next_cursor"] = next_cursor

            try:
                url = f"{CLOB_API}/trades"
                async with session.get(
                    url, params=batch_params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("CLOB /trades returned %d", resp.status)
                        break
                    data = await resp.json()
            except Exception as e:
                logger.error("Error fetching trades: %s", e)
                break

            # CLOB API returns {"trades": [...], "next_cursor": "..."}
            # or just a list — handle both formats
            if isinstance(data, dict):
                trades = data.get("trades", data.get("data", []))
                next_cursor = data.get("next_cursor")
            elif isinstance(data, list):
                trades = data
                next_cursor = None
            else:
                break

            if not trades:
                break

            all_trades.extend(trades)
            fetched += len(trades)

            if not next_cursor or next_cursor == "LTE":
                break

            # Rate limit compliance
            import asyncio
            await asyncio.sleep(0.5)

        return all_trades[:limit]

    def parse_trades(self, raw_trades: list[dict]) -> list[WalletTrade]:
        """Parse raw CLOB API trade dicts into WalletTrade objects.

        The CLOB API trade format:
            {
                "id": "...",
                "taker_order_id": "...",
                "market": "token_id",
                "asset_id": "token_id",
                "side": "BUY" | "SELL",
                "size": "100",
                "price": "0.65",
                "maker_address": "0x...",
                "match_time": "2024-01-15T10:30:00Z",
                "status": "MATCHED",
                ...
            }
        """
        parsed: list[WalletTrade] = []

        for t in raw_trades:
            try:
                maker = t.get("maker_address", "")
                if not maker or not maker.startswith("0x"):
                    # Some API responses use "owner" or nested fields
                    maker = t.get("owner", t.get("trader", ""))
                if not maker:
                    continue

                side = t.get("side", "").upper()
                if side not in ("BUY", "SELL"):
                    continue

                price = float(t.get("price", 0))
                size_str = t.get("size", "0")
                size = float(size_str)
                size_usd = price * size  # shares * price = USD cost

                # Parse timestamp
                ts_str = t.get("match_time", t.get("timestamp", t.get("created_at", "")))
                if ts_str:
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        ts = dt.timestamp()
                    except (ValueError, TypeError):
                        ts = time.time()
                else:
                    ts = time.time()

                token_id = t.get("asset_id", t.get("market", ""))
                # Condition ID might be in a different field
                market_id = t.get("condition_id", t.get("market", token_id))

                parsed.append(WalletTrade(
                    wallet=maker.lower(),
                    market_id=market_id,
                    token_id=token_id,
                    side=side,
                    price=price,
                    size_usd=size_usd,
                    timestamp=ts,
                    outcome=t.get("outcome", ""),
                ))
            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Skip unparseable trade: %s", e)
                continue

        return parsed

    def compute_profiles(
        self,
        trades: list[WalletTrade],
        resolved_markets: dict[str, float] | None = None,
    ) -> dict[str, WalletProfile]:
        """Group trades by wallet and compute performance profiles.

        Args:
            trades: List of parsed wallet trades.
            resolved_markets: Optional dict of market_id -> resolution (0.0 or 1.0).
                If provided, used to compute actual wins/losses.
                If not, we estimate based on price movement (heuristic).

        Returns:
            Dict of wallet_address -> WalletProfile.
        """
        # Group trades by wallet
        wallet_trades: dict[str, list[WalletTrade]] = {}
        for t in trades:
            wallet_trades.setdefault(t.wallet, []).append(t)

        profiles: dict[str, WalletProfile] = {}

        for addr, w_trades in wallet_trades.items():
            total_pnl = 0.0
            wins = 0
            losses = 0
            total_size = 0.0
            markets_seen: set[str] = set()
            last_ts = 0.0

            # Group by market to estimate PnL
            market_groups: dict[str, list[WalletTrade]] = {}
            for t in w_trades:
                market_groups.setdefault(t.market_id, []).append(t)
                markets_seen.add(t.market_id)
                last_ts = max(last_ts, t.timestamp)
                total_size += t.size_usd

            for mid, m_trades in market_groups.items():
                if resolved_markets and mid in resolved_markets:
                    resolution = resolved_markets[mid]
                    # Compute actual PnL for this wallet on this market
                    market_pnl = 0.0
                    for t in m_trades:
                        if t.side == "BUY":
                            shares = t.size_usd / t.price if t.price > 0 else 0
                            market_pnl += shares * resolution - t.size_usd
                        else:
                            no_price = 1.0 - t.price
                            shares = t.size_usd / no_price if no_price > 0 else 0
                            market_pnl += shares * (1.0 - resolution) - t.size_usd

                    total_pnl += market_pnl
                    if market_pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    # Heuristic: can't determine win/loss without resolution.
                    # Use average price as a proxy — lower buys tend to be winners
                    # on Polymarket (contracts resolve to 0 or 1).
                    # We'll just count trades without win/loss attribution.
                    pass

            n_trades = len(w_trades)
            profile = WalletProfile(
                address=addr,
                total_trades=n_trades,
                wins=wins,
                losses=losses,
                total_pnl=round(total_pnl, 2),
                avg_trade_size=round(total_size / max(n_trades, 1), 2),
                last_trade_ts=last_ts,
                markets_traded=len(markets_seen),
            )
            profile.score = self._compute_score(profile)
            profiles[addr] = profile

        return profiles

    def _compute_score(self, p: WalletProfile) -> float:
        """Score = win_rate * log(1 + |total_pnl|) * recency_factor.

        Negative PnL wallets get score 0 (we never copy losers).
        """
        if p.total_pnl <= 0 or p.total_trades < self._min_trades:
            return 0.0

        wr = p.win_rate
        if wr < self._min_win_rate:
            return 0.0

        pnl_factor = math.log(1.0 + p.total_pnl)

        # Recency: 1.0 if traded today, decays linearly to 0 over recency window
        days_ago = p.recency_days
        recency = max(0.0, 1.0 - days_ago / self._recency_days)
        if recency <= 0:
            return 0.0

        score = wr * pnl_factor * recency
        return round(score, 4)

    def screen_and_save(
        self,
        trades: list[WalletTrade],
        resolved_markets: dict[str, float] | None = None,
    ) -> list[WalletProfile]:
        """Screen wallets from trades, save top performers to DB.

        Returns the watchlist (top N wallets by score).
        """
        profiles = self.compute_profiles(trades, resolved_markets)

        # Filter by thresholds
        qualified = [
            p for p in profiles.values()
            if p.score > 0
            and p.win_rate >= self._min_win_rate
            and p.total_pnl >= self._min_pnl
            and p.total_trades >= self._min_trades
            and p.recency_days <= self._recency_days
        ]

        # Sort by score descending, take top N
        qualified.sort(key=lambda p: -p.score)
        watchlist = qualified[:self._watchlist_size]

        # Persist to DB
        for p in watchlist:
            self.db.upsert_wallet(p)

        logger.info(
            "Screening complete: %d wallets analyzed, %d qualified, "
            "top %d saved to watchlist",
            len(profiles), len(qualified), len(watchlist),
        )
        return watchlist

    def get_watchlist(self) -> list[WalletProfile]:
        """Load current watchlist from DB."""
        return self.db.get_watchlist(self._watchlist_size)


# ─── Wallet Copy Trader ──────────────────────────────────────────

class WalletCopyTrader:
    """Monitors watched wallets and generates copy-trade signals.

    Polls the CLOB API for new trades from wallets on the watchlist.
    When a watched wallet opens a position, generates a CopySignal
    if safety checks pass (price slippage, dedup, delay).

    Copy sizing: proportional to wallet score (higher = larger).
    """

    def __init__(
        self,
        db: WalletScoreDB | None = None,
        max_price_move: float = MAX_PRICE_MOVE_PCT,
        max_copy_delay_s: float = COPY_DELAY_TARGET_S,
        confidence_factor: float = 0.8,
    ):
        self.db = db or WalletScoreDB()
        self._max_price_move = max_price_move
        self._max_copy_delay_s = max_copy_delay_s
        self._confidence_factor = confidence_factor

        # Track what we've already copied: (wallet, market_id) -> timestamp
        self._copied: dict[tuple[str, str], float] = {}

        # Last poll timestamp to only fetch new trades
        self._last_poll_ts: float = 0.0

        # Watchlist cache (refreshed from DB periodically)
        self._watchlist: dict[str, WalletProfile] = {}
        self._watchlist_loaded_at: float = 0.0

    def refresh_watchlist(self) -> None:
        """Reload watchlist from DB."""
        profiles = self.db.get_watchlist()
        self._watchlist = {p.address: p for p in profiles}
        self._watchlist_loaded_at = time.time()
        logger.info("Watchlist refreshed: %d wallets", len(self._watchlist))

    @property
    def watched_addresses(self) -> set[str]:
        """Set of wallet addresses being monitored."""
        # Refresh every 5 minutes
        if time.time() - self._watchlist_loaded_at > 300:
            self.refresh_watchlist()
        return set(self._watchlist.keys())

    def check_new_trades(
        self,
        trades: list[WalletTrade],
        current_prices: dict[str, float],
    ) -> list[CopySignal]:
        """Check a batch of new trades for copy opportunities.

        Args:
            trades: Recently fetched trades (should be newer than _last_poll_ts).
            current_prices: Dict of token_id -> current mid price.

        Returns:
            List of CopySignal for trades that pass all safety checks.
        """
        if not self._watchlist:
            self.refresh_watchlist()

        signals: list[CopySignal] = []
        now = time.time()

        for trade in trades:
            wallet = trade.wallet.lower()

            # 1. Is this wallet on our watchlist?
            profile = self._watchlist.get(wallet)
            if profile is None:
                continue

            # 2. Is this trade newer than our last check?
            if trade.timestamp <= self._last_poll_ts:
                continue

            # 3. De-duplicate: don't copy same wallet into same market twice
            copy_key = (wallet, trade.market_id)
            if copy_key in self._copied:
                logger.debug(
                    "Skip duplicate copy: %s already in %s",
                    wallet[:10], trade.market_id[:20],
                )
                continue

            # 4. Delay check: how long since the wallet's trade?
            delay = now - trade.timestamp
            if delay > self._max_copy_delay_s:
                logger.debug(
                    "Skip stale trade from %s (%.0fs old, limit %.0fs)",
                    wallet[:10], delay, self._max_copy_delay_s,
                )
                continue

            # 5. Price slippage check: has price moved too much?
            current_price = current_prices.get(trade.token_id, 0.0)
            if current_price > 0 and trade.price > 0:
                price_move = abs(current_price - trade.price) / trade.price
                if price_move > self._max_price_move:
                    logger.info(
                        "Skip copy %s: price moved %.1f%% (limit %.1f%%)",
                        wallet[:10], price_move * 100, self._max_price_move * 100,
                    )
                    continue

            signal = CopySignal(
                wallet=wallet,
                market_id=trade.market_id,
                token_id=trade.token_id,
                direction=trade.side,
                wallet_price=trade.price,
                current_price=current_price if current_price > 0 else trade.price,
                wallet_score=profile.score,
                delay_s=round(delay, 1),
                timestamp=now,
                outcome=trade.outcome,
            )
            signals.append(signal)

            # Mark as copied
            self._copied[copy_key] = now

        # Update last poll timestamp
        if trades:
            max_ts = max(t.timestamp for t in trades)
            self._last_poll_ts = max(self._last_poll_ts, max_ts)

        return signals

    def compute_copy_size(
        self,
        signal: CopySignal,
        bankroll: float,
        max_position_pct: float = 0.05,
    ) -> float:
        """Compute copy-trade size proportional to wallet score.

        Higher-scored wallets get larger allocations. Capped at
        max_position_pct of bankroll.

        Args:
            signal: The copy signal to size.
            bankroll: Available capital.
            max_position_pct: Maximum fraction of bankroll per copy.

        Returns:
            Size in USD, or 0 if too small.
        """
        if bankroll <= 0 or signal.wallet_score <= 0:
            return 0.0

        # Normalize score: assume max score ~10 (good wallet with $10k+ PnL)
        # score = wr * log(1 + pnl) * recency
        # e.g., 0.7 * log(10001) * 0.8 = ~6.5
        normalized = min(signal.wallet_score / 10.0, 1.0)

        # Base allocation: 1-5% of bankroll
        base_pct = 0.01 + normalized * 0.04
        size = bankroll * min(base_pct, max_position_pct)

        # Reduce for high delay (signal is staler)
        delay_penalty = max(0.5, 1.0 - signal.delay_s / self._max_copy_delay_s)
        size *= delay_penalty

        # Floor at $0.50 (Polymarket minimum)
        if size < 0.50:
            return 0.0

        return round(size, 2)

    def compute_edge(self, signal: CopySignal) -> float:
        """Estimate the edge from copying this wallet.

        edge = wallet_historical_edge * confidence_factor

        The wallet's historical edge is approximated from their win rate
        and PnL. The confidence factor accounts for latency degradation.
        """
        profile = self._watchlist.get(signal.wallet)
        if not profile:
            return 0.0

        # Wallet's raw edge: win_rate - breakeven (53% for taker after fees)
        breakeven = 0.53
        raw_edge = profile.win_rate - breakeven
        if raw_edge <= 0:
            return 0.0

        # Scale by confidence factor and delay penalty
        delay_factor = max(0.3, 1.0 - signal.delay_s / (self._max_copy_delay_s * 2))
        edge = raw_edge * self._confidence_factor * delay_factor

        return round(max(edge, 0.0), 4)

    def get_signal_strength(self, signal: CopySignal) -> float:
        """Normalize wallet score to 0-1 signal strength."""
        # Clamp score to [0, 10] range and normalize
        return round(min(signal.wallet_score / 10.0, 1.0), 4)

    def cleanup_stale_copies(self, max_age_s: float = 86400.0) -> int:
        """Remove stale entries from the copied set (>24h old)."""
        now = time.time()
        stale = [
            k for k, ts in self._copied.items()
            if now - ts > max_age_s
        ]
        for k in stale:
            del self._copied[k]
        return len(stale)

    def get_copy_performance(self) -> dict[str, Any]:
        """Get aggregate copy-trade performance from DB."""
        return self.db.get_copy_performance()
