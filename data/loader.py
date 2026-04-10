"""Unified backtest data loader for Polymarket.

Provides a single entry point to load market metadata + price history
from multiple sources with automatic fallback:

    1. Local SQLite cache (fastest)
    2. CLOB API price history (best for recent markets)
    3. PMXT orderbook snapshots (best for older/purged markets)
    4. Gamma API metadata (always available)

Returns standardized MarketData objects ready for backtesting.

Usage:
    from data.loader import BacktestLoader

    loader = BacktestLoader()
    markets = loader.load_markets(min_count=500, min_volume=1000)
    for m in markets:
        print(m.condition_id, m.question, len(m.price_series))
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "historical.db"
PMXT_DIR = DATA_DIR / "pmxt_orderbooks"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


# ────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────


@dataclass
class PricePoint:
    """Single price observation."""
    timestamp: int          # Unix epoch seconds
    price: float            # 0-1 probability
    source: str = "clob"    # "clob", "pmxt", "synthetic"


@dataclass
class OrderbookSnapshot:
    """Single orderbook snapshot for a market."""
    timestamp: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bid_depth: float
    ask_depth: float


@dataclass
class MarketData:
    """Standardized market data for backtesting."""
    condition_id: str
    question: str
    category: str
    slug: str
    resolution: str             # "resolved", "closed", "unknown"
    outcome: str                # Final outcome string
    tokens: list[str]           # CLOB token IDs [YES, NO]
    volume: float
    liquidity: float
    created_at: str
    closed_at: str
    end_date: str
    tags: list[str]
    price_series: list[PricePoint] = field(default_factory=list)
    orderbook_snapshots: list[OrderbookSnapshot] = field(default_factory=list)
    price_source: str = "none"  # Where prices came from

    @property
    def has_prices(self) -> bool:
        return len(self.price_series) > 0

    @property
    def price_df(self) -> pd.DataFrame:
        """Convert price series to DataFrame."""
        if not self.price_series:
            return pd.DataFrame(columns=["timestamp", "price", "source"])
        rows = [{"timestamp": p.timestamp, "price": p.price, "source": p.source}
                for p in self.price_series]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON/storage)."""
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "category": self.category,
            "resolution": self.resolution,
            "outcome": self.outcome,
            "volume": self.volume,
            "n_prices": len(self.price_series),
            "n_snapshots": len(self.orderbook_snapshots),
            "price_source": self.price_source,
        }


# ────────────────────────────────────────────────────────────
# SQLite cache layer
# ────────────────────────────────────────────────────────────


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Open or create the historical database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Ensure tables exist (idempotent)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT,
            slug TEXT,
            category TEXT,
            end_date TEXT,
            resolution TEXT,
            outcome TEXT,
            outcome_prices TEXT,
            tokens TEXT,
            tags TEXT,
            liquidity REAL,
            volume REAL,
            created_at TEXT,
            closed_at TEXT,
            resolved_at TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT,
            token_id TEXT,
            timestamp INTEGER,
            price REAL,
            source TEXT DEFAULT 'clob',
            UNIQUE(condition_id, token_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS orderbook_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT,
            timestamp INTEGER,
            best_bid REAL,
            best_ask REAL,
            mid_price REAL,
            spread REAL,
            bid_depth REAL,
            ask_depth REAL,
            UNIQUE(condition_id, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_ph_cond ON price_history(condition_id);
        CREATE INDEX IF NOT EXISTS idx_ph_ts ON price_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_ob_cond ON orderbook_cache(condition_id);
    """)
    conn.commit()
    return conn


def _safe_json_loads(val: str | None) -> list | dict:
    if not val:
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


# ────────────────────────────────────────────────────────────
# Gamma API — market metadata
# ────────────────────────────────────────────────────────────


def _fetch_gamma_markets(
    limit: int = 1000,
    offset: int = 0,
    closed_only: bool = True,
    min_volume: float = 0,
) -> list[dict[str, Any]]:
    """Fetch markets from Gamma API with pagination."""
    all_markets: list[dict[str, Any]] = []
    page_size = min(limit, 100)
    current_offset = offset
    remaining = limit

    while remaining > 0:
        params: dict[str, Any] = {
            "limit": min(page_size, remaining),
            "offset": current_offset,
            "order": "volume",
            "ascending": "false",
        }
        if closed_only:
            params["closed"] = "true"

        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets", params=params, timeout=30,
                headers={"User-Agent": "polymarket-backtester/1.0"},
            )
            resp.raise_for_status()
            markets = resp.json()

            if not markets:
                break

            # Filter by minimum volume
            if min_volume > 0:
                markets = [m for m in markets
                           if float(m.get("volume", m.get("volumeNum", 0)) or 0) >= min_volume]

            all_markets.extend(markets)
            remaining -= page_size
            current_offset += page_size

            logger.info("Gamma API: fetched %d markets (total: %d)", len(markets), len(all_markets))
            time.sleep(0.5)

        except requests.RequestException as e:
            logger.error("Gamma API error at offset %d: %s", current_offset, e)
            break

    return all_markets


def _store_market(conn: sqlite3.Connection, m: dict[str, Any]) -> bool:
    """Store a single market in SQLite. Returns True if successful."""
    try:
        tokens = m.get("clobTokenIds", "[]")
        if isinstance(tokens, str):
            tokens = _safe_json_loads(tokens)
        elif not isinstance(tokens, list):
            tokens = []

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = _safe_json_loads(prices)

        tags = m.get("tags", [])
        if isinstance(tags, str):
            tags = _safe_json_loads(tags)

        resolution = "unknown"
        if m.get("resolved"):
            resolution = "resolved"
        elif m.get("closed"):
            resolution = "closed"

        category = m.get("category", "")
        if not category and tags:
            category = tags[0] if isinstance(tags, list) and tags else "other"

        conn.execute(
            """INSERT OR REPLACE INTO markets
               (condition_id, question, slug, category, end_date,
                resolution, outcome, outcome_prices, tokens, tags,
                liquidity, volume, created_at, closed_at, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                m.get("conditionId", m.get("condition_id", "")),
                m.get("question", ""),
                m.get("slug", ""),
                category,
                m.get("endDate", ""),
                resolution,
                m.get("outcome", ""),
                json.dumps(prices),
                json.dumps(tokens),
                json.dumps(tags),
                float(m.get("liquidity", 0) or 0),
                float(m.get("volume", m.get("volumeNum", 0)) or 0),
                m.get("createdAt", ""),
                m.get("closedTime", ""),
                (m.get("description", "") or "")[:500],
            ),
        )
        return True
    except Exception as e:
        logger.debug("Failed to store market: %s", e)
        return False


# ────────────────────────────────────────────────────────────
# CLOB API — price history
# ────────────────────────────────────────────────────────────


def _fetch_clob_prices(
    token_id: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch price history from CLOB API for a single token."""
    params: dict[str, Any] = {"market": token_id}
    if start_ts:
        params["startTs"] = start_ts
    if end_ts:
        params["endTs"] = end_ts

    try:
        resp = requests.get(
            f"{CLOB_BASE}/prices-history",
            params=params,
            timeout=30,
            headers={"User-Agent": "polymarket-backtester/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])
    except requests.RequestException as e:
        logger.debug("CLOB API error for %s: %s", token_id[:20], e)
        return []


def _store_prices(
    conn: sqlite3.Connection,
    condition_id: str,
    token_id: str,
    history: list[dict[str, Any]],
    source: str = "clob",
) -> int:
    """Store price history in SQLite. Returns count stored."""
    stored = 0
    for point in history:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO price_history
                   (condition_id, token_id, timestamp, price, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    condition_id,
                    token_id,
                    int(point.get("t", 0)),
                    float(point.get("p", 0)),
                    source,
                ),
            )
            stored += 1
        except Exception:
            pass
    conn.commit()
    return stored


# ────────────────────────────────────────────────────────────
# PMXT — orderbook snapshot fallback
# ────────────────────────────────────────────────────────────


def _load_pmxt_prices_for_market(
    condition_id: str,
    pmxt_dir: Path = PMXT_DIR,
) -> list[PricePoint]:
    """Extract price series for a market from cached PMXT Parquet files.

    This is used as fallback when CLOB API has no price history.
    """
    if not pmxt_dir.exists():
        return []

    parquet_files = sorted(pmxt_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    # Lazy import to avoid circular dependency
    from data.pmxt_parser import iter_row_groups, _detect_columns

    prices: list[PricePoint] = []

    for fp in parquet_files:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(fp))
            sample = pf.read_row_group(0).to_pandas().head(5)
            col_map = _detect_columns(sample)

            market_col = col_map.get("market_id")
            if not market_col:
                continue

            # Read needed columns only
            needed = {market_col}
            if "timestamp" in col_map:
                needed.add(col_map["timestamp"])
            if "price" in col_map:
                needed.add(col_map["price"])
            elif "best_bid" in col_map and "best_ask" in col_map:
                needed.add(col_map["best_bid"])
                needed.add(col_map["best_ask"])

            available = {f.name for f in pf.schema_arrow}
            read_cols = sorted(needed & available)

            for chunk in iter_row_groups(fp, columns=read_cols):
                filtered = chunk[chunk[market_col] == condition_id]
                if filtered.empty:
                    continue

                for _, row in filtered.iterrows():
                    ts_val = row.get(col_map.get("timestamp", ""), 0)
                    if hasattr(ts_val, "timestamp"):
                        ts = int(ts_val.timestamp())
                    else:
                        ts = int(ts_val) if ts_val else 0

                    if "price" in col_map and col_map["price"] in row.index:
                        price = float(row[col_map["price"]])
                    elif "best_bid" in col_map:
                        bid = float(row.get(col_map.get("best_bid", ""), 0) or 0)
                        ask = float(row.get(col_map.get("best_ask", ""), 0) or 0)
                        price = (bid + ask) / 2.0 if bid and ask else 0
                    else:
                        continue

                    if 0 < price < 1 and ts > 0:
                        prices.append(PricePoint(timestamp=ts, price=price, source="pmxt"))

        except Exception as e:
            logger.debug("PMXT parse error for %s in %s: %s", condition_id[:20], fp.name, e)

    prices.sort(key=lambda p: p.timestamp)
    return prices


# ────────────────────────────────────────────────────────────
# Main loader class
# ────────────────────────────────────────────────────────────


class BacktestLoader:
    """Unified data loader with caching and source fallback.

    Usage:
        loader = BacktestLoader()
        markets = loader.load_markets(min_count=500)
        for m in markets:
            print(m.condition_id, len(m.price_series))
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = _init_db(db_path)
        logger.info("BacktestLoader initialized (db=%s)", db_path)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Market metadata ──

    def get_cached_market_count(self) -> int:
        """How many markets are in the local DB."""
        row = self.conn.execute("SELECT COUNT(*) FROM markets").fetchone()
        return row[0] if row else 0

    def refresh_markets(
        self,
        target_count: int = 600,
        min_volume: float = 0,
    ) -> int:
        """Ensure we have at least target_count markets in the DB.

        Fetches from Gamma API if needed.
        Returns total market count.
        """
        current = self.get_cached_market_count()
        if current >= target_count:
            logger.info("Already have %d markets (target: %d)", current, target_count)
            return current

        need = target_count - current
        logger.info("Need %d more markets (have %d, target %d)", need, current, target_count)

        markets = _fetch_gamma_markets(
            limit=need + 100,  # Fetch extra in case some fail to store
            offset=current,
            min_volume=min_volume,
        )

        stored = 0
        for m in markets:
            if _store_market(self.conn, m):
                stored += 1
        self.conn.commit()

        total = self.get_cached_market_count()
        logger.info("Stored %d new markets (total now: %d)", stored, total)
        return total

    def _load_market_metadata(
        self,
        limit: int = 1000,
        min_volume: float = 0,
        category: str | None = None,
        resolved_only: bool = True,
    ) -> list[MarketData]:
        """Load market metadata from local SQLite cache."""
        query = "SELECT * FROM markets WHERE 1=1"
        params: list[Any] = []

        if resolved_only:
            query += " AND resolution IN ('resolved', 'closed')"
        if min_volume > 0:
            query += " AND volume >= ?"
            params.append(min_volume)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY volume DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        markets = []
        for row in rows:
            data = dict(zip(columns, row))
            tokens = _safe_json_loads(data.get("tokens", "[]"))
            tags = _safe_json_loads(data.get("tags", "[]"))
            if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
                tokens = [t.get("token_id", "") for t in tokens]

            markets.append(MarketData(
                condition_id=data.get("condition_id", ""),
                question=data.get("question", ""),
                category=data.get("category", ""),
                slug=data.get("slug", ""),
                resolution=data.get("resolution", "unknown"),
                outcome=data.get("outcome", ""),
                tokens=tokens if isinstance(tokens, list) else [],
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                created_at=data.get("created_at", ""),
                closed_at=data.get("closed_at", ""),
                end_date=data.get("end_date", ""),
                tags=tags if isinstance(tags, list) else [],
            ))

        return markets

    # ── Price history ──

    def _load_cached_prices(self, condition_id: str) -> list[PricePoint]:
        """Load price history from SQLite cache."""
        cursor = self.conn.execute(
            """SELECT timestamp, price, COALESCE(source, 'clob') as source
               FROM price_history
               WHERE condition_id = ?
               ORDER BY timestamp""",
            (condition_id,),
        )
        return [
            PricePoint(timestamp=row[0], price=row[1], source=row[2])
            for row in cursor.fetchall()
        ]

    def _load_cached_orderbook(self, condition_id: str) -> list[OrderbookSnapshot]:
        """Load orderbook snapshots from SQLite cache."""
        cursor = self.conn.execute(
            """SELECT timestamp, best_bid, best_ask, mid_price, spread,
                      bid_depth, ask_depth
               FROM orderbook_cache
               WHERE condition_id = ?
               ORDER BY timestamp""",
            (condition_id,),
        )
        return [
            OrderbookSnapshot(
                timestamp=row[0], best_bid=row[1], best_ask=row[2],
                mid_price=row[3], spread=row[4], bid_depth=row[5], ask_depth=row[6],
            )
            for row in cursor.fetchall()
        ]

    def fetch_prices(
        self,
        market: MarketData,
        use_cache: bool = True,
        try_clob: bool = True,
        try_pmxt: bool = True,
    ) -> MarketData:
        """Fetch price history for a market using fallback chain.

        Order: cache -> CLOB API -> PMXT snapshots
        """
        # 1. Check cache first
        if use_cache:
            cached = self._load_cached_prices(market.condition_id)
            if cached:
                market.price_series = cached
                market.price_source = "cache"
                return market

        # 2. Try CLOB API
        if try_clob and market.tokens:
            for token_id in market.tokens[:1]:  # Just the YES token
                if not token_id or not isinstance(token_id, str):
                    continue

                history = _fetch_clob_prices(token_id)
                if history:
                    _store_prices(self.conn, market.condition_id, token_id, history, "clob")
                    market.price_series = [
                        PricePoint(
                            timestamp=int(p.get("t", 0)),
                            price=float(p.get("p", 0)),
                            source="clob",
                        )
                        for p in history
                    ]
                    market.price_source = "clob"
                    return market

                time.sleep(0.3)

        # 3. Fallback to PMXT orderbook snapshots
        if try_pmxt:
            pmxt_prices = _load_pmxt_prices_for_market(market.condition_id)
            if pmxt_prices:
                # Cache PMXT-derived prices
                for pp in pmxt_prices:
                    try:
                        self.conn.execute(
                            """INSERT OR IGNORE INTO price_history
                               (condition_id, token_id, timestamp, price, source)
                               VALUES (?, ?, ?, ?, ?)""",
                            (market.condition_id, "pmxt", pp.timestamp, pp.price, "pmxt"),
                        )
                    except Exception:
                        pass
                self.conn.commit()
                market.price_series = pmxt_prices
                market.price_source = "pmxt"
                return market

        market.price_source = "none"
        return market

    # ── Batch loading ──

    def load_markets(
        self,
        min_count: int = 500,
        min_volume: float = 0,
        category: str | None = None,
        fetch_prices: bool = True,
        max_price_fetches: int = 500,
        clob_sleep: float = 0.3,
    ) -> list[MarketData]:
        """Load markets with metadata and prices, ensuring minimum count.

        This is the main entry point. Steps:
        1. Ensure enough markets in local DB (fetch from Gamma if needed)
        2. Load metadata from DB
        3. Fetch price histories (cache -> CLOB -> PMXT fallback)
        4. Return standardized MarketData objects

        Args:
            min_count: Minimum number of markets to return.
            min_volume: Minimum lifetime volume filter.
            category: Optional category filter.
            fetch_prices: Whether to fetch price histories.
            max_price_fetches: Max CLOB API calls for prices.
            clob_sleep: Delay between CLOB API calls.

        Returns:
            List of MarketData with price series attached.
        """
        # Step 1: Ensure we have enough markets
        self.refresh_markets(target_count=min_count + 100, min_volume=min_volume)

        # Step 2: Load metadata
        markets = self._load_market_metadata(
            limit=min_count + 200,  # extra buffer for filtering
            min_volume=min_volume,
            category=category,
        )
        logger.info("Loaded %d market metadata records", len(markets))

        if not fetch_prices:
            return markets[:min_count]

        # Step 3: Fetch prices with fallback chain
        fetched = 0
        for i, market in enumerate(markets):
            self.fetch_prices(market, use_cache=True, try_clob=True, try_pmxt=True)

            if market.price_source == "clob":
                fetched += 1
                time.sleep(clob_sleep)

            if fetched >= max_price_fetches:
                # Stop making API calls, but continue loading from cache
                for remaining in markets[i + 1:]:
                    self.fetch_prices(remaining, use_cache=True, try_clob=False, try_pmxt=True)
                break

            if (i + 1) % 50 == 0:
                with_prices = sum(1 for m in markets[:i + 1] if m.has_prices)
                logger.info(
                    "Progress: %d/%d markets loaded, %d with prices, %d CLOB fetches",
                    i + 1, len(markets), with_prices, fetched,
                )

        # Step 4: Report results
        with_prices = sum(1 for m in markets if m.has_prices)
        by_source = {}
        for m in markets:
            by_source[m.price_source] = by_source.get(m.price_source, 0) + 1

        logger.info(
            "Loaded %d markets: %d with prices. Sources: %s",
            len(markets), with_prices, by_source,
        )

        return markets[:min_count]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the local data store."""
        stats: dict[str, Any] = {}

        row = self.conn.execute("SELECT COUNT(*) FROM markets").fetchone()
        stats["total_markets"] = row[0] if row else 0

        row = self.conn.execute(
            "SELECT COUNT(*) FROM markets WHERE resolution = 'resolved'"
        ).fetchone()
        stats["resolved_markets"] = row[0] if row else 0

        row = self.conn.execute("SELECT COUNT(*) FROM price_history").fetchone()
        stats["total_price_points"] = row[0] if row else 0

        row = self.conn.execute(
            "SELECT COUNT(DISTINCT condition_id) FROM price_history"
        ).fetchone()
        stats["markets_with_prices"] = row[0] if row else 0

        # Category breakdown
        rows = self.conn.execute(
            "SELECT category, COUNT(*) FROM markets GROUP BY category ORDER BY COUNT(*) DESC LIMIT 20"
        ).fetchall()
        stats["categories"] = {r[0]: r[1] for r in rows}

        # PMXT files
        if PMXT_DIR.exists():
            pmxt_files = list(PMXT_DIR.glob("*.parquet"))
            stats["pmxt_files"] = len(pmxt_files)
            stats["pmxt_size_mb"] = round(
                sum(f.stat().st_size for f in pmxt_files) / (1024 * 1024), 1
            )
        else:
            stats["pmxt_files"] = 0
            stats["pmxt_size_mb"] = 0

        return stats


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Unified backtest data loader")
    parser.add_argument("--load", type=int, default=0,
                        help="Load N markets with prices")
    parser.add_argument("--stats", action="store_true",
                        help="Print database statistics")
    parser.add_argument("--refresh", type=int, default=0,
                        help="Ensure at least N markets in DB")
    parser.add_argument("--min-volume", type=float, default=0,
                        help="Minimum market volume filter")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="SQLite database path")
    args = parser.parse_args()

    with BacktestLoader(db_path=Path(args.db)) as loader:
        if args.stats:
            stats = loader.get_stats()
            print(json.dumps(stats, indent=2))
            return

        if args.refresh > 0:
            total = loader.refresh_markets(
                target_count=args.refresh,
                min_volume=args.min_volume,
            )
            print(f"Database now has {total} markets")
            return

        if args.load > 0:
            markets = loader.load_markets(
                min_count=args.load,
                min_volume=args.min_volume,
                category=args.category,
            )
            print(f"\nLoaded {len(markets)} markets:")
            with_prices = [m for m in markets if m.has_prices]
            print(f"  With prices: {len(with_prices)}")

            # Print summary table
            print(f"\n{'Question':<50} {'Vol':>10} {'Prices':>8} {'Source':>8}")
            print(f"{'-'*50} {'-'*10} {'-'*8} {'-'*8}")
            for m in markets[:20]:
                q = m.question[:48] if m.question else "?"
                print(f"{q:<50} {m.volume:>10.0f} {len(m.price_series):>8} {m.price_source:>8}")

            if len(markets) > 20:
                print(f"  ... and {len(markets) - 20} more")

            return

        # Default: show stats
        stats = loader.get_stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
