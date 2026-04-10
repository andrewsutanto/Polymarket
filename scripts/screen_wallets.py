#!/usr/bin/env python3
"""Standalone wallet screening script — build initial watchlist.

Fetches recent trades from the Polymarket CLOB API, groups by wallet
address, computes performance metrics, and saves the top performers
to SQLite for copy-trading.

Usage:
    python scripts/screen_wallets.py
    python scripts/screen_wallets.py --min-trades 50 --min-winrate 0.60
    python scripts/screen_wallets.py --limit 2000 --top 30

The script will:
    1. Fetch recent trades from CLOB API (across active markets)
    2. Group by wallet (maker) address
    3. Compute win rate, PnL, trade count for each wallet
    4. Filter by thresholds
    5. Save top N to SQLite (data/wallet_scores.db)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import aiohttp

from core.wallet_tracker import (
    WalletScreener,
    WalletScoreDB,
    WalletTrade,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


async def fetch_active_token_ids(session: aiohttp.ClientSession, limit: int = 50) -> list[str]:
    """Fetch token IDs from the top active markets."""
    try:
        async with session.get(
            f"{GAMMA_API}/markets",
            params={"active": "true", "closed": "false", "limit": limit,
                    "order": "volume24hr", "ascending": "false"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                logger.error("Gamma API returned %d", resp.status)
                return []
            markets = await resp.json()
    except Exception as e:
        logger.error("Error fetching markets: %s", e)
        return []

    token_ids: list[str] = []
    for m in markets:
        tokens_raw = m.get("clobTokenIds", "[]")
        if isinstance(tokens_raw, str):
            try:
                tids = json.loads(tokens_raw)
            except (json.JSONDecodeError, TypeError):
                continue
        elif isinstance(tokens_raw, list):
            tids = tokens_raw
        else:
            continue
        token_ids.extend(str(t) for t in tids)

    return token_ids


async def fetch_trades_for_token(
    session: aiohttp.ClientSession,
    token_id: str,
    limit: int = 200,
) -> list[dict]:
    """Fetch recent trades for a single token from CLOB API."""
    try:
        async with session.get(
            f"{CLOB_API}/trades",
            params={"asset_id": token_id},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception as e:
        logger.debug("Error fetching trades for %s: %s", token_id[:20], e)
        return []

    if isinstance(data, dict):
        return data.get("trades", data.get("data", []))[:limit]
    elif isinstance(data, list):
        return data[:limit]
    return []


async def main():
    parser = argparse.ArgumentParser(
        description="Screen Polymarket wallets and build copy-trade watchlist",
    )
    parser.add_argument(
        "--min-trades", type=int, default=50,
        help="Minimum trades per wallet (default: 50)",
    )
    parser.add_argument(
        "--min-winrate", type=float, default=0.60,
        help="Minimum win rate (default: 0.60)",
    )
    parser.add_argument(
        "--min-pnl", type=float, default=1000.0,
        help="Minimum total PnL in USD (default: 1000)",
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Max total trades to fetch (default: 1000)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top wallets to save (default: 20)",
    )
    parser.add_argument(
        "--markets", type=int, default=30,
        help="Number of active markets to scan (default: 30)",
    )
    parser.add_argument(
        "--db", type=str, default="data/wallet_scores.db",
        help="SQLite database path (default: data/wallet_scores.db)",
    )
    args = parser.parse_args()

    logger.info("=== Polymarket Wallet Screener ===")
    logger.info("  Min trades: %d", args.min_trades)
    logger.info("  Min win rate: %.0f%%", args.min_winrate * 100)
    logger.info("  Min PnL: $%.0f", args.min_pnl)
    logger.info("  Fetch limit: %d trades", args.limit)
    logger.info("  Top wallets: %d", args.top)
    logger.info("  Markets to scan: %d", args.markets)

    db = WalletScoreDB(args.db)
    screener = WalletScreener(
        db=db,
        min_win_rate=args.min_winrate,
        min_pnl=args.min_pnl,
        min_trades=args.min_trades,
        watchlist_size=args.top,
    )

    async with aiohttp.ClientSession() as session:
        # Step 1: Get active market token IDs
        logger.info("Fetching active markets...")
        token_ids = await fetch_active_token_ids(session, limit=args.markets)
        logger.info("Found %d token IDs across %d markets", len(token_ids), args.markets)

        if not token_ids:
            logger.error("No token IDs found. Check API connectivity.")
            return

        # Step 2: Fetch trades for each market
        all_raw_trades: list[dict] = []
        trades_per_token = max(1, args.limit // len(token_ids))

        for i, tid in enumerate(token_ids):
            if len(all_raw_trades) >= args.limit:
                break

            raw = await fetch_trades_for_token(session, tid, limit=trades_per_token)
            all_raw_trades.extend(raw)

            if (i + 1) % 10 == 0:
                logger.info(
                    "  Fetched %d/%d tokens, %d trades so far...",
                    i + 1, len(token_ids), len(all_raw_trades),
                )

            # Rate limit: don't exceed 1 request per 0.5s
            await asyncio.sleep(0.5)

        logger.info("Total raw trades fetched: %d", len(all_raw_trades))

        if not all_raw_trades:
            logger.error(
                "No trades returned. The CLOB API may not expose wallet "
                "addresses in this endpoint. Consider using Polygon RPC "
                "or Dune Analytics as alternative data sources."
            )
            return

    # Step 3: Parse and screen
    logger.info("Parsing trades...")
    parsed_trades = screener.parse_trades(all_raw_trades)
    logger.info("Parsed %d trades with wallet addresses", len(parsed_trades))

    if not parsed_trades:
        logger.error(
            "No trades had wallet addresses. The CLOB /trades endpoint "
            "may not include maker_address. Alternatives:\n"
            "  1. Use Polygon RPC to read CTF Exchange Transfer events\n"
            "  2. Use Dune Analytics: SELECT * FROM polymarket.trades\n"
            "  3. Check if /activity endpoint exposes addresses"
        )
        return

    # Store raw trades for future analysis
    logger.info("Storing trades to DB...")
    for t in parsed_trades:
        db.insert_trade(t)

    # Step 4: Screen wallets
    logger.info("Screening wallets...")
    # Note: Without resolved market data, we estimate using heuristics.
    # For production, feed in resolved_markets from Gamma API.
    watchlist = screener.screen_and_save(parsed_trades)

    # Step 5: Print results
    if watchlist:
        logger.info("\n=== Top %d Wallets ===", len(watchlist))
        print(f"\n{'Rank':<5} {'Address':<15} {'Trades':<8} {'WR':<8} "
              f"{'PnL':<12} {'Score':<8} {'Last Trade':<12}")
        print("-" * 75)

        for i, w in enumerate(watchlist, 1):
            days_ago = w.recency_days
            recency = f"{days_ago:.1f}d ago" if days_ago < 30 else ">30d"
            addr_short = f"{w.address[:6]}...{w.address[-4:]}"
            print(
                f"{i:<5} {addr_short:<15} {w.total_trades:<8} "
                f"{w.win_rate:.1%}   ${w.total_pnl:>9,.2f}  "
                f"{w.score:<8.2f} {recency:<12}"
            )

        print(f"\nWatchlist saved to: {args.db}")
    else:
        logger.warning(
            "No wallets passed screening thresholds. Try lowering:\n"
            "  --min-trades %d  (currently %d)\n"
            "  --min-winrate %.2f  (currently %.2f)\n"
            "  --min-pnl %.0f  (currently %.0f)",
            max(10, args.min_trades // 2), args.min_trades,
            max(0.50, args.min_winrate - 0.05), args.min_winrate,
            max(100, args.min_pnl / 2), args.min_pnl,
        )

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
