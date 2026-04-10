"""Historical Polymarket data downloader for backtesting.

Sources (in priority order):
1. Gamma API — all resolved markets with metadata (free, no auth)
2. CLOB API — price history timeseries per market (free, no auth)
3. HuggingFace — SII-WANGZJ/Polymarket_data (1.1B trade records)

Usage:
    python data/download_historical.py --source gamma --limit 1000
    python data/download_historical.py --source clob --markets data/resolved_markets.json
    python data/download_historical.py --source huggingface --dataset trades
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "historical.db"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for historical data."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

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
            UNIQUE(condition_id, token_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            timestamp INTEGER,
            maker TEXT,
            taker TEXT,
            UNIQUE(condition_id, token_id, timestamp, price, size)
        );

        CREATE INDEX IF NOT EXISTS idx_ph_cond ON price_history(condition_id);
        CREATE INDEX IF NOT EXISTS idx_ph_ts ON price_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trades_cond ON trades(condition_id);
        CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
    """)
    conn.commit()
    return conn


def fetch_resolved_markets(
    limit: int = 1000,
    offset: int = 0,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch all resolved markets from Gamma API.

    The Gamma API supports pagination and filtering.
    """
    all_markets: list[dict[str, Any]] = []
    page_size = min(limit, 100)  # API max per page
    current_offset = offset
    remaining = limit

    while remaining > 0:
        params: dict[str, Any] = {
            "limit": min(page_size, remaining),
            "offset": current_offset,
            "closed": "true",
            "order": "volume",
            "ascending": "false",
        }
        if category:
            params["tag"] = category

        try:
            resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
            resp.raise_for_status()
            markets = resp.json()

            if not markets:
                break

            all_markets.extend(markets)
            remaining -= len(markets)
            current_offset += len(markets)

            logger.info(
                "Fetched %d markets (total: %d/%d)",
                len(markets), len(all_markets), limit
            )

            # Rate limiting
            time.sleep(0.5)

        except requests.RequestException as e:
            logger.error("Gamma API error at offset %d: %s", current_offset, e)
            break

    return all_markets


def store_markets(conn: sqlite3.Connection, markets: list[dict[str, Any]]) -> int:
    """Store market metadata in SQLite."""
    stored = 0
    for m in markets:
        try:
            # Parse token IDs and prices
            tokens = m.get("clobTokenIds", "[]")
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except json.JSONDecodeError:
                    tokens = []

            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except json.JSONDecodeError:
                    prices = []

            tags = m.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError:
                    tags = []

            # Determine resolution outcome
            resolution = "unknown"
            if m.get("resolved"):
                resolution = "resolved"
            elif m.get("closed"):
                resolution = "closed"

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
                    m.get("category", m.get("tags", ["other"])[0] if tags else "other"),
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
                    m.get("description", "")[:500],
                ),
            )
            stored += 1
        except Exception as e:
            logger.warning("Failed to store market %s: %s", m.get("question", "?")[:40], e)

    conn.commit()
    logger.info("Stored %d markets in DB", stored)
    return stored


def fetch_price_history(
    conn: sqlite3.Connection,
    condition_id: str,
    token_id: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch price history for a token from CLOB API and store in DB."""
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
        )
        resp.raise_for_status()
        data = resp.json()

        history = data.get("history", [])
        if not history:
            return []

        # Store in DB
        for point in history:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO price_history
                       (condition_id, token_id, timestamp, price)
                       VALUES (?, ?, ?, ?)""",
                    (
                        condition_id,
                        token_id,
                        int(point.get("t", 0)),
                        float(point.get("p", 0)),
                    ),
                )
            except Exception:
                pass

        conn.commit()
        return history

    except requests.RequestException as e:
        logger.warning("CLOB price history error for %s: %s", token_id[:20], e)
        return []


def fetch_all_price_histories(
    conn: sqlite3.Connection,
    max_markets: int = 500,
    sleep_between: float = 0.3,
) -> int:
    """Fetch price histories for all stored markets."""
    cursor = conn.execute(
        """SELECT condition_id, tokens FROM markets
           WHERE tokens != '[]' AND tokens != ''
           ORDER BY volume DESC
           LIMIT ?""",
        (max_markets,),
    )
    rows = cursor.fetchall()

    total_points = 0
    for i, (cid, tokens_json) in enumerate(rows):
        try:
            tokens = json.loads(tokens_json) if isinstance(tokens_json, str) else tokens_json
        except json.JSONDecodeError:
            continue

        if not tokens:
            continue

        for token_id in tokens[:2]:  # YES and NO tokens
            if isinstance(token_id, dict):
                token_id = token_id.get("token_id", "")
            if not token_id:
                continue

            history = fetch_price_history(conn, cid, token_id)
            total_points += len(history)
            time.sleep(sleep_between)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d markets, %d price points", i + 1, len(rows), total_points)

    logger.info("Fetched %d total price points for %d markets", total_points, len(rows))
    return total_points


def get_db_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get statistics about the local database."""
    stats = {}

    row = conn.execute("SELECT COUNT(*) FROM markets").fetchone()
    stats["total_markets"] = row[0] if row else 0

    row = conn.execute("SELECT COUNT(*) FROM markets WHERE resolution = 'resolved'").fetchone()
    stats["resolved_markets"] = row[0] if row else 0

    row = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()
    stats["price_points"] = row[0] if row else 0

    row = conn.execute("SELECT COUNT(DISTINCT condition_id) FROM price_history").fetchone()
    stats["markets_with_prices"] = row[0] if row else 0

    row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
    stats["total_trades"] = row[0] if row else 0

    # Category breakdown
    rows = conn.execute(
        "SELECT category, COUNT(*) FROM markets GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    stats["categories"] = {r[0]: r[1] for r in rows}

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Polymarket historical data")
    parser.add_argument("--source", choices=["gamma", "clob", "stats"], default="gamma")
    parser.add_argument("--limit", type=int, default=1000, help="Max markets to fetch")
    parser.add_argument("--category", type=str, default=None, help="Filter by category tag")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="SQLite DB path")
    args = parser.parse_args()

    conn = init_db(Path(args.db))

    if args.source == "gamma":
        logger.info("Fetching resolved markets from Gamma API (limit=%d)...", args.limit)
        markets = fetch_resolved_markets(limit=args.limit, category=args.category)
        stored = store_markets(conn, markets)
        logger.info("Done: fetched %d, stored %d markets", len(markets), stored)

    elif args.source == "clob":
        logger.info("Fetching price histories from CLOB API...")
        points = fetch_all_price_histories(conn, max_markets=args.limit)
        logger.info("Done: %d price points fetched", points)

    elif args.source == "stats":
        stats = get_db_stats(conn)
        print(json.dumps(stats, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
