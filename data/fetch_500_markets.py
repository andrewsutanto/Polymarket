"""Fetch 500+ resolved Polymarket markets from HuggingFace for backtesting.

Uses SII-WANGZJ/Polymarket_data on HuggingFace (1.1B+ trade records):
  - markets.parquet (85MB, 538K markets) -- metadata + resolution
  - quant.parquet (28GB, 418M trades)    -- unified YES-perspective trades

Strategy:
  1. Download markets.parquet (small, fits in memory)
  2. Filter to resolved markets with sufficient volume
  3. Store metadata in SQLite
  4. Stream quant.parquet in row-group chunks
  5. Build OHLC price timeseries from trades (1-hour bars)
  6. Store price history in SQLite

Falls back to direct HTTP download if `datasets` library is unavailable.

Usage:
    # Full pipeline: markets + price histories (recommended)
    python data/fetch_500_markets.py

    # Metadata only (fast, ~85MB download)
    python data/fetch_500_markets.py --metadata-only

    # Target more markets
    python data/fetch_500_markets.py --target 1000

    # Check database status
    python data/fetch_500_markets.py --status

    # Resume price fetch for markets that have metadata but no prices
    python data/fetch_500_markets.py --prices-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "historical.db"
CACHE_DIR = DATA_DIR / "hf_cache"

HF_DATASET = "SII-WANGZJ/Polymarket_data"
HF_BASE_URL = (
    "https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data"
    "/resolve/main"
)

# Minimum trades per market to build a useful price series
MIN_TRADES_FOR_PRICES = 20
# Resample interval for price bars
PRICE_BAR_INTERVAL = "1h"


# ────────────────────────────────────────────────────────────
# Database setup (matches existing schema in download_historical.py)
# ────────────────────────────────────────────────────────────


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create or open the historical database with all needed tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Larger cache for bulk inserts
    conn.execute("PRAGMA cache_size=-64000")  # 64MB

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
            source TEXT DEFAULT 'hf_quant',
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
        CREATE INDEX IF NOT EXISTS idx_markets_vol ON markets(volume);
    """)
    conn.commit()
    return conn


# ────────────────────────────────────────────────────────────
# HuggingFace data download helpers
# ────────────────────────────────────────────────────────────


def _ensure_pandas():
    """Import pandas, raising helpful error if missing."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        logger.error("pandas is required: pip install pandas pyarrow")
        sys.exit(1)


def _download_hf_file(filename: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download a single file from the HuggingFace dataset via HTTP.

    Supports resuming partial downloads.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename

    url = f"{HF_BASE_URL}/{filename}"

    # Check if already downloaded
    if local_path.exists():
        local_size = local_path.stat().st_size
        # Verify size with HEAD request
        try:
            head = requests.head(url, timeout=30, allow_redirects=True)
            remote_size = int(head.headers.get("content-length", 0))
            if remote_size > 0 and local_size >= remote_size:
                logger.info("Already downloaded: %s (%.1f MB)", filename,
                            local_size / (1024 * 1024))
                return local_path
            elif local_size > 0:
                logger.info("Resuming download of %s from %.1f MB",
                            filename, local_size / (1024 * 1024))
        except Exception:
            if local_size > 1024 * 1024:  # >1MB, probably valid
                return local_path

    # Download with resume support
    headers = {}
    mode = "wb"
    existing_size = 0
    if local_path.exists():
        existing_size = local_path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    logger.info("Downloading %s from HuggingFace...", filename)

    try:
        resp = requests.get(url, stream=True, timeout=120,
                            headers=headers, allow_redirects=True)

        if resp.status_code == 416:
            # Range not satisfiable -- file is complete
            return local_path

        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + existing_size
        downloaded = existing_size
        last_log = time.time()

        with open(local_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log >= 10:
                    pct = (downloaded / total * 100) if total else 0
                    mb = downloaded / (1024 * 1024)
                    logger.info("  %s: %.1f MB / %.1f MB (%.0f%%)",
                                filename, mb, total / (1024 * 1024), pct)
                    last_log = now

        size_mb = downloaded / (1024 * 1024)
        logger.info("Downloaded %s (%.1f MB)", filename, size_mb)
        return local_path

    except Exception as e:
        logger.error("Failed to download %s: %s", filename, e)
        raise


def _load_parquet_file(path: Path):
    """Load a parquet file, trying pyarrow first, then fastparquet."""
    pd = _ensure_pandas()
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception:
            return pd.read_parquet(path)


def _iter_parquet_row_groups(
    path: Path,
    columns: list[str] | None = None,
    batch_size: int = 500_000,
) -> Iterator:
    """Stream a large parquet file in row-group chunks.

    Yields pandas DataFrames, one per row group (or batched).
    """
    pd = _ensure_pandas()

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        n_groups = pf.metadata.num_row_groups
        logger.info("Parquet file has %d row groups, %d total rows",
                     n_groups, pf.metadata.num_rows)

        for i in range(n_groups):
            table = pf.read_row_group(i, columns=columns)
            df = table.to_pandas()
            yield df

            if (i + 1) % 10 == 0:
                logger.info("  Processed row group %d / %d", i + 1, n_groups)

    except ImportError:
        logger.warning("pyarrow not available, loading full file (slow)...")
        df = pd.read_parquet(path, columns=columns)
        # Yield in chunks
        for start in range(0, len(df), batch_size):
            yield df.iloc[start:start + batch_size]


# ────────────────────────────────────────────────────────────
# Step 1: Load and store market metadata
# ────────────────────────────────────────────────────────────


def load_hf_markets(
    target: int = 600,
    min_volume: float = 0,
    cache_dir: Path = CACHE_DIR,
) -> "pd.DataFrame":
    """Download and filter markets.parquet from HuggingFace.

    Returns DataFrame of resolved markets sorted by volume descending.
    """
    pd = _ensure_pandas()

    # Try `datasets` library first (handles caching/streaming natively)
    markets_df = None
    try:
        from datasets import load_dataset
        logger.info("Loading markets via `datasets` library...")
        ds = load_dataset(HF_DATASET, data_files="markets.parquet", split="train")
        markets_df = ds.to_pandas()
        logger.info("Loaded %d markets via datasets library", len(markets_df))
    except ImportError:
        logger.info("`datasets` library not available, using direct HTTP download")
    except Exception as e:
        logger.warning("datasets library failed (%s), falling back to HTTP", e)

    # Fallback: direct HTTP download
    if markets_df is None:
        markets_path = _download_hf_file("markets.parquet", cache_dir)
        markets_df = _load_parquet_file(markets_path)
        logger.info("Loaded %d markets from parquet file", len(markets_df))

    # Log column info
    logger.info("Markets columns: %s", list(markets_df.columns))

    # Filter to resolved/closed markets
    if "closed" in markets_df.columns:
        resolved = markets_df[markets_df["closed"] == 1].copy()
        logger.info("Resolved/closed markets: %d", len(resolved))
    else:
        resolved = markets_df.copy()
        logger.warning("No 'closed' column found, using all markets")

    # Filter by volume
    if "volume" in resolved.columns:
        if min_volume > 0:
            resolved = resolved[resolved["volume"] >= min_volume]
            logger.info("Markets with volume >= %.0f: %d", min_volume, len(resolved))

        # Sort by volume descending (highest quality first)
        resolved = resolved.sort_values("volume", ascending=False)
    else:
        logger.warning("No 'volume' column, cannot sort by volume")

    # Take top N
    result = resolved.head(target).copy()
    logger.info("Selected top %d markets by volume", len(result))

    return result


def store_hf_markets(
    conn: sqlite3.Connection,
    markets_df: "pd.DataFrame",
) -> int:
    """Store HuggingFace market metadata into the SQLite schema.

    Maps HF columns to existing schema:
      HF                -> SQLite
      id/market_id      -> (not stored, used as join key for trades)
      condition_id      -> condition_id
      question          -> question
      slug              -> slug
      token1            -> tokens[0]
      token2            -> tokens[1]
      closed            -> resolution
      outcome_prices    -> outcome_prices, outcome
      volume            -> volume
      event_title       -> category (fallback)
      created_at        -> created_at
      end_date          -> end_date, closed_at
    """
    pd = _ensure_pandas()
    stored = 0
    skipped = 0

    for _, row in markets_df.iterrows():
        try:
            # Extract condition_id (primary key)
            cid = str(row.get("condition_id", ""))
            if not cid or cid == "nan" or cid == "None":
                skipped += 1
                continue

            # Build tokens list from token1/token2
            tokens = []
            for tk in ["token1", "token2"]:
                val = row.get(tk, "")
                if val and str(val) not in ("nan", "None", ""):
                    tokens.append(str(val))

            # Determine resolution
            closed_val = row.get("closed", 0)
            if closed_val == 1 or closed_val is True:
                resolution = "resolved"
            else:
                resolution = "closed" if row.get("archived", 0) else "unknown"

            # Parse outcome_prices (may be a list, numpy array, or JSON string)
            outcome_prices_raw = row.get("outcome_prices", "")
            if isinstance(outcome_prices_raw, (list, tuple)):
                prices_list = [str(x) for x in outcome_prices_raw]
                outcome_prices = json.dumps(prices_list)
            elif hasattr(outcome_prices_raw, "tolist"):  # numpy array
                prices_list = [str(x) for x in outcome_prices_raw.tolist()]
                outcome_prices = json.dumps(prices_list)
            elif isinstance(outcome_prices_raw, str) and outcome_prices_raw:
                outcome_prices = outcome_prices_raw
                try:
                    prices_list = json.loads(outcome_prices)
                except (json.JSONDecodeError, ValueError):
                    prices_list = []
            else:
                outcome_prices = "[]"
                prices_list = []

            # Determine outcome from outcome_prices
            outcome = ""
            try:
                if isinstance(prices_list, list) and len(prices_list) >= 2:
                    # The token with price closest to 1.0 won
                    p1 = float(prices_list[0]) if prices_list[0] else 0
                    p2 = float(prices_list[1]) if prices_list[1] else 0
                    a1 = str(row.get("answer1", "Yes"))
                    a2 = str(row.get("answer2", "No"))
                    if p1 > p2:
                        outcome = a1
                    elif p2 > p1:
                        outcome = a2
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

            # Category: use event_title or slug-based heuristic
            category = ""
            event_title = str(row.get("event_title", ""))
            if event_title and event_title not in ("nan", "None"):
                # Use first word of event title as rough category
                category = event_title
            slug = str(row.get("slug", ""))

            # Volume
            vol = float(row.get("volume", 0) or 0)

            # Dates (may be pandas Timestamps or strings)
            def _fmt_date(val) -> str:
                if val is None:
                    return ""
                s = str(val)
                if s in ("nan", "None", "NaT", "NaT+00:00", ""):
                    return ""
                # pandas Timestamps have isoformat()
                if hasattr(val, "isoformat"):
                    return val.isoformat()
                return s

            created = _fmt_date(row.get("created_at"))
            end_date = _fmt_date(row.get("end_date"))

            conn.execute(
                """INSERT OR REPLACE INTO markets
                   (condition_id, question, slug, category, end_date,
                    resolution, outcome, outcome_prices, tokens, tags,
                    liquidity, volume, created_at, closed_at, resolved_at,
                    description)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    str(row.get("question", "")),
                    slug if slug not in ("nan", "None") else "",
                    category[:200] if category else "",
                    end_date,
                    resolution,
                    outcome,
                    outcome_prices,
                    json.dumps(tokens),
                    json.dumps([]),  # No tags from HF dataset
                    0.0,  # No liquidity in HF dataset
                    vol,
                    created,
                    end_date,  # Use end_date as closed_at
                    end_date,  # Use end_date as resolved_at
                    "",        # No description in HF dataset
                ),
            )
            stored += 1

        except Exception as e:
            logger.debug("Failed to store market %s: %s",
                         str(row.get("question", "?"))[:40], e)
            skipped += 1

    conn.commit()
    logger.info("Stored %d markets, skipped %d", stored, skipped)
    return stored


# ────────────────────────────────────────────────────────────
# Step 2: Build price timeseries from HF trade data
# ────────────────────────────────────────────────────────────


def _get_markets_needing_prices(conn: sqlite3.Connection, limit: int = 2000) -> dict[str, str]:
    """Get condition_ids of markets that need price history.

    Returns dict mapping condition_id -> market_id (for joining with trades).
    Also loads the HF market_id mapping.
    """
    # Get all stored condition_ids that lack price data
    cursor = conn.execute("""
        SELECT m.condition_id
        FROM markets m
        LEFT JOIN (
            SELECT condition_id, COUNT(*) as cnt
            FROM price_history
            GROUP BY condition_id
        ) ph ON m.condition_id = ph.condition_id
        WHERE m.resolution IN ('resolved', 'closed')
          AND (ph.cnt IS NULL OR ph.cnt < 10)
        ORDER BY m.volume DESC
        LIMIT ?
    """, (limit,))

    return {row[0] for row in cursor.fetchall()}


def build_price_series_from_quant(
    conn: sqlite3.Connection,
    target_condition_ids: set[str],
    cache_dir: Path = CACHE_DIR,
    max_chunk_markets: int = 5000,
) -> dict[str, int]:
    """Stream quant.parquet and build hourly price bars for target markets.

    quant.parquet schema (unified YES perspective):
      timestamp, block_number, transaction_hash, log_index,
      market_id, condition_id, event_id,
      price, usd_amount, token_amount, side, maker, taker

    We aggregate into 1-hour VWAP bars and store as price_history.

    Returns stats dict.
    """
    pd = _ensure_pandas()
    import numpy as np

    if not target_condition_ids:
        logger.info("No markets need price history")
        return {"markets_processed": 0, "total_points": 0}

    logger.info("Building price series for %d markets from quant.parquet",
                len(target_condition_ids))

    # Try datasets library first
    quant_path = None
    try:
        from datasets import load_dataset
        logger.info("Trying `datasets` library for quant.parquet (streaming)...")
        # Streaming mode to avoid downloading the full 28GB
        ds = load_dataset(
            HF_DATASET,
            data_files="quant.parquet",
            split="train",
            streaming=True,
        )
        # Process streaming dataset
        return _process_streaming_quant(conn, ds, target_condition_ids)
    except ImportError:
        logger.info("`datasets` not available, using direct download")
    except Exception as e:
        logger.warning("datasets streaming failed (%s), falling back to HTTP", e)

    # Fallback: download quant.parquet and process in row-group chunks
    quant_path = _download_hf_file("quant.parquet", cache_dir)
    return _process_parquet_quant(conn, quant_path, target_condition_ids)


def _process_streaming_quant(
    conn: sqlite3.Connection,
    dataset,
    target_cids: set[str],
) -> dict[str, int]:
    """Process quant data from HuggingFace streaming dataset."""
    pd = _ensure_pandas()
    import numpy as np

    stats = {"markets_processed": 0, "total_points": 0, "rows_scanned": 0}

    # Accumulate trades per market
    # Dict[condition_id -> list of (timestamp, price, usd_amount)]
    market_trades: dict[str, list] = {}
    batch_size = 100_000
    batch = []

    for row in dataset:
        stats["rows_scanned"] += 1
        cid = row.get("condition_id", "")
        if cid not in target_cids:
            continue

        ts = int(row.get("timestamp", 0))
        price = float(row.get("price", 0))
        usd = float(row.get("usd_amount", 0))

        if 0 < price <= 1 and ts > 0:
            if cid not in market_trades:
                market_trades[cid] = []
            market_trades[cid].append((ts, price, usd))

        # Periodically flush markets that have enough data
        if stats["rows_scanned"] % 1_000_000 == 0:
            logger.info("  Scanned %dM rows, tracking %d markets",
                        stats["rows_scanned"] // 1_000_000, len(market_trades))

            # Flush markets with enough trades
            flush_stats = _flush_market_trades(conn, market_trades, min_trades=50)
            stats["markets_processed"] += flush_stats["flushed"]
            stats["total_points"] += flush_stats["points"]

    # Final flush
    flush_stats = _flush_market_trades(
        conn, market_trades, min_trades=MIN_TRADES_FOR_PRICES, force_all=True
    )
    stats["markets_processed"] += flush_stats["flushed"]
    stats["total_points"] += flush_stats["points"]

    logger.info(
        "Streaming complete: scanned %d rows, %d markets, %d price points",
        stats["rows_scanned"], stats["markets_processed"], stats["total_points"],
    )
    return stats


def _process_parquet_quant(
    conn: sqlite3.Connection,
    quant_path: Path,
    target_cids: set[str],
) -> dict[str, int]:
    """Process quant.parquet file in row-group chunks."""
    pd = _ensure_pandas()

    stats = {"markets_processed": 0, "total_points": 0, "rows_scanned": 0}

    # Only read the columns we need
    needed_cols = ["condition_id", "timestamp", "price", "usd_amount"]

    # Accumulate trades per market across chunks
    market_trades: dict[str, list] = {}

    for chunk_df in _iter_parquet_row_groups(quant_path, columns=needed_cols):
        stats["rows_scanned"] += len(chunk_df)

        # Filter to target markets
        mask = chunk_df["condition_id"].isin(target_cids)
        relevant = chunk_df[mask]

        if relevant.empty:
            continue

        # Accumulate trades
        for cid, group in relevant.groupby("condition_id"):
            trades_list = list(zip(
                group["timestamp"].astype(int),
                group["price"].astype(float),
                group["usd_amount"].astype(float),
            ))
            if cid not in market_trades:
                market_trades[cid] = []
            market_trades[cid].extend(trades_list)

        # Log progress
        logger.info(
            "  Scanned %dM rows, tracking %d markets with trades",
            stats["rows_scanned"] // 1_000_000,
            len(market_trades),
        )

        # Flush markets that have accumulated enough trades (>500)
        # to avoid excessive memory use
        flush_stats = _flush_market_trades(
            conn, market_trades, min_trades=500, keep_threshold=500
        )
        stats["markets_processed"] += flush_stats["flushed"]
        stats["total_points"] += flush_stats["points"]

    # Final flush of all remaining markets
    flush_stats = _flush_market_trades(
        conn, market_trades, min_trades=MIN_TRADES_FOR_PRICES, force_all=True
    )
    stats["markets_processed"] += flush_stats["flushed"]
    stats["total_points"] += flush_stats["points"]

    logger.info(
        "Parquet processing complete: %d rows, %d markets, %d price points",
        stats["rows_scanned"], stats["markets_processed"], stats["total_points"],
    )
    return stats


def _flush_market_trades(
    conn: sqlite3.Connection,
    market_trades: dict[str, list],
    min_trades: int = 20,
    keep_threshold: int = 0,
    force_all: bool = False,
) -> dict[str, int]:
    """Convert accumulated raw trades to hourly price bars and store in DB.

    Args:
        market_trades: Dict mapping condition_id -> [(ts, price, usd), ...]
        min_trades: Minimum trades to consider a market worth storing.
        keep_threshold: Only flush markets with MORE than this many trades
                       (to allow more accumulation). Ignored if force_all.
        force_all: Flush everything regardless of count.

    Returns dict with flush stats. Mutates market_trades by removing flushed.
    """
    pd = _ensure_pandas()
    import numpy as np

    stats = {"flushed": 0, "points": 0}

    to_remove = []

    for cid, trades in market_trades.items():
        n = len(trades)

        # Skip if not enough trades and not forcing
        if not force_all and keep_threshold > 0 and n <= keep_threshold:
            continue
        if n < min_trades:
            if force_all:
                to_remove.append(cid)
            continue

        # Build DataFrame from accumulated trades
        df = pd.DataFrame(trades, columns=["timestamp", "price", "usd_amount"])

        # Convert to datetime for resampling
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("datetime")

        # Compute VWAP per hour
        df["weighted_price"] = df["price"] * df["usd_amount"].clip(lower=0.01)
        hourly = df.set_index("datetime").resample("1h").agg({
            "weighted_price": "sum",
            "usd_amount": "sum",
            "price": ["first", "last", "min", "max", "count"],
            "timestamp": "first",
        })

        # Flatten multi-level columns
        hourly.columns = [
            "wp_sum", "usd_sum",
            "open", "close", "low", "high", "trade_count",
            "ts_first",
        ]

        # Filter to hours with actual trades
        hourly = hourly[hourly["trade_count"] > 0]

        if len(hourly) < 3:
            to_remove.append(cid)
            continue

        # VWAP price, falling back to simple close
        hourly["vwap"] = np.where(
            hourly["usd_sum"] > 0,
            hourly["wp_sum"] / hourly["usd_sum"],
            hourly["close"],
        )
        # Clamp to valid probability range
        hourly["vwap"] = hourly["vwap"].clip(0.001, 0.999)

        # Store in price_history table
        points_stored = 0
        for idx, bar in hourly.iterrows():
            ts = int(bar["ts_first"])
            price = float(bar["vwap"])
            if ts > 0 and 0 < price < 1:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO price_history
                           (condition_id, token_id, timestamp, price, source)
                           VALUES (?, ?, ?, ?, 'hf_quant')""",
                        (cid, "yes", ts, price),
                    )
                    points_stored += 1
                except Exception:
                    pass

        conn.commit()
        stats["flushed"] += 1
        stats["points"] += points_stored
        to_remove.append(cid)

    # Clean up flushed entries
    for cid in to_remove:
        market_trades.pop(cid, None)

    if stats["flushed"] > 0:
        logger.info("  Flushed %d markets (%d price points)",
                     stats["flushed"], stats["points"])

    return stats


# ────────────────────────────────────────────────────────────
# Alternative: Build prices from trades.parquet (original semantics)
# ────────────────────────────────────────────────────────────


def build_price_series_from_trades(
    conn: sqlite3.Connection,
    target_condition_ids: set[str],
    cache_dir: Path = CACHE_DIR,
) -> dict[str, int]:
    """Alternative: use trades.parquet instead of quant.parquet.

    trades.parquet has the original trade semantics with maker/taker
    direction. We use the price directly since it's already 0-1.

    Use this if quant.parquet download is too large.
    """
    pd = _ensure_pandas()

    if not target_condition_ids:
        return {"markets_processed": 0, "total_points": 0}

    trades_path = _download_hf_file("trades.parquet", cache_dir)

    stats = {"markets_processed": 0, "total_points": 0, "rows_scanned": 0}
    needed_cols = ["condition_id", "timestamp", "price", "usd_amount"]
    market_trades: dict[str, list] = {}

    for chunk_df in _iter_parquet_row_groups(trades_path, columns=needed_cols):
        stats["rows_scanned"] += len(chunk_df)
        mask = chunk_df["condition_id"].isin(target_condition_ids)
        relevant = chunk_df[mask]

        if relevant.empty:
            continue

        for cid, group in relevant.groupby("condition_id"):
            trades_list = list(zip(
                group["timestamp"].astype(int),
                group["price"].astype(float),
                group["usd_amount"].astype(float),
            ))
            if cid not in market_trades:
                market_trades[cid] = []
            market_trades[cid].extend(trades_list)

        logger.info("  Scanned %dM rows, %d markets",
                     stats["rows_scanned"] // 1_000_000, len(market_trades))

        flush_stats = _flush_market_trades(
            conn, market_trades, min_trades=500, keep_threshold=500
        )
        stats["markets_processed"] += flush_stats["flushed"]
        stats["total_points"] += flush_stats["points"]

    flush_stats = _flush_market_trades(
        conn, market_trades, min_trades=MIN_TRADES_FOR_PRICES, force_all=True
    )
    stats["markets_processed"] += flush_stats["flushed"]
    stats["total_points"] += flush_stats["points"]

    return stats


# ────────────────────────────────────────────────────────────
# Status and reporting
# ────────────────────────────────────────────────────────────


def print_status(conn: sqlite3.Connection) -> None:
    """Print detailed database status."""
    print("\n" + "=" * 65)
    print("  Polymarket Backtest Database Status (HuggingFace Pipeline)")
    print("=" * 65)

    # Market counts
    row = conn.execute("SELECT COUNT(*) FROM markets").fetchone()
    total = row[0] if row else 0

    row = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE resolution = 'resolved'"
    ).fetchone()
    resolved = row[0] if row else 0

    row = conn.execute(
        "SELECT COUNT(*) FROM markets WHERE resolution = 'closed'"
    ).fetchone()
    closed = row[0] if row else 0

    print(f"\n  Markets: {total} total ({resolved} resolved, {closed} closed)")

    # Volume stats
    row = conn.execute(
        "SELECT SUM(volume), AVG(volume), MAX(volume) FROM markets WHERE volume > 0"
    ).fetchone()
    if row and row[0]:
        print(f"  Volume:  ${row[0]:,.0f} total | ${row[1]:,.0f} avg | ${row[2]:,.0f} max")

    # Price history
    row = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()
    total_pts = row[0] if row else 0

    row = conn.execute(
        "SELECT COUNT(DISTINCT condition_id) FROM price_history"
    ).fetchone()
    markets_with_prices = row[0] if row else 0

    print(f"\n  Price history: {total_pts:,} points across {markets_with_prices} markets")

    if total > 0:
        pct = markets_with_prices / total * 100
        print(f"  Coverage: {pct:.1f}% of markets have price data")

    # Source breakdown
    rows = conn.execute("""
        SELECT COALESCE(source, 'unknown'), COUNT(DISTINCT condition_id), COUNT(*)
        FROM price_history
        GROUP BY source
    """).fetchall()
    if rows:
        print(f"\n  {'Source':<15} {'Markets':>8} {'Points':>10}")
        print(f"  {'-'*15} {'-'*8} {'-'*10}")
        for src, mkt_cnt, pt_cnt in rows:
            print(f"  {src:<15} {mkt_cnt:>8} {pt_cnt:>10,}")

    # Price points per market distribution
    rows = conn.execute("""
        SELECT
            CASE
                WHEN cnt >= 500 THEN '500+'
                WHEN cnt >= 100 THEN '100-499'
                WHEN cnt >= 50  THEN '50-99'
                WHEN cnt >= 20  THEN '20-49'
                ELSE '<20'
            END as bucket,
            COUNT(*) as n_markets
        FROM (
            SELECT condition_id, COUNT(*) as cnt
            FROM price_history
            GROUP BY condition_id
        )
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '500+' THEN 1
                WHEN '100-499' THEN 2
                WHEN '50-99' THEN 3
                WHEN '20-49' THEN 4
                ELSE 5
            END
    """).fetchall()
    if rows:
        print(f"\n  {'Price Points/Market':<20} {'Count':>8}")
        print(f"  {'-'*20} {'-'*8}")
        for bucket, cnt in rows:
            print(f"  {bucket:<20} {cnt:>8}")

    # Readiness assessment
    print(f"\n  {'='*63}")
    ready_count = 0
    if markets_with_prices > 0:
        row = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT condition_id
                FROM price_history
                GROUP BY condition_id
                HAVING COUNT(*) >= 20
            )
        """).fetchone()
        ready_count = row[0] if row else 0

    if ready_count >= 500:
        print(f"  READY: {ready_count} markets with 20+ price points (target: 500)")
    elif ready_count >= 200:
        print(f"  PARTIAL: {ready_count} markets with 20+ price points (target: 500)")
        print(f"  Sufficient for OOS validation (need 200+)")
    else:
        print(f"  NOT READY: {ready_count} markets with 20+ price points (need 500)")

    print("  " + "=" * 63 + "\n")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fetch 500+ Polymarket markets from HuggingFace for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/fetch_500_markets.py                     # Full pipeline
  python data/fetch_500_markets.py --metadata-only     # Just market metadata (fast)
  python data/fetch_500_markets.py --prices-only       # Build prices for existing markets
  python data/fetch_500_markets.py --status            # Check database status
  python data/fetch_500_markets.py --target 1000       # Get 1000 markets
  python data/fetch_500_markets.py --use-trades        # Use trades.parquet instead of quant
        """,
    )
    parser.add_argument("--target", type=int, default=600,
                        help="Target number of markets (default: 600)")
    parser.add_argument("--min-volume", type=float, default=0,
                        help="Minimum market volume filter (USD)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only fetch market metadata, skip price histories")
    parser.add_argument("--prices-only", action="store_true",
                        help="Only build prices for markets already in DB")
    parser.add_argument("--use-trades", action="store_true",
                        help="Use trades.parquet instead of quant.parquet for prices")
    parser.add_argument("--status", action="store_true",
                        help="Print database status and exit")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="SQLite database path")
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR),
                        help="Directory for cached HuggingFace downloads")
    args = parser.parse_args()

    db_path = Path(args.db)
    cache_dir = Path(args.cache_dir)
    conn = init_db(db_path)

    if args.status:
        print_status(conn)
        conn.close()
        return

    start_time = time.time()

    # Step 1: Market metadata
    if not args.prices_only:
        logger.info("=" * 60)
        logger.info("STEP 1: Loading market metadata from HuggingFace")
        logger.info("=" * 60)

        markets_df = load_hf_markets(
            target=args.target,
            min_volume=args.min_volume,
            cache_dir=cache_dir,
        )

        stored = store_hf_markets(conn, markets_df)
        logger.info("Stored %d markets in %s", stored, db_path)

        # We need the market_id -> condition_id mapping for trades
        # Store it for price building
        if "id" in markets_df.columns:
            market_id_map = dict(zip(
                markets_df["id"].astype(str),
                markets_df["condition_id"].astype(str),
            ))
            logger.info("Market ID mapping: %d entries", len(market_id_map))
    else:
        logger.info("Skipping metadata (--prices-only)")

    # Step 2: Price histories
    if not args.metadata_only:
        logger.info("=" * 60)
        logger.info("STEP 2: Building price histories from HuggingFace trades")
        logger.info("=" * 60)

        target_cids = _get_markets_needing_prices(conn, limit=args.target)
        logger.info("Markets needing prices: %d", len(target_cids))

        if target_cids:
            if args.use_trades:
                logger.info("Using trades.parquet (28GB)")
                price_stats = build_price_series_from_trades(
                    conn, target_cids, cache_dir,
                )
            else:
                logger.info("Using quant.parquet (28GB, unified YES perspective)")
                price_stats = build_price_series_from_quant(
                    conn, target_cids, cache_dir,
                )

            logger.info("Price build stats: %s", price_stats)
        else:
            logger.info("All markets already have price data!")
    else:
        logger.info("Skipping price histories (--metadata-only)")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)

    # Final status
    print_status(conn)
    conn.close()


if __name__ == "__main__":
    main()
