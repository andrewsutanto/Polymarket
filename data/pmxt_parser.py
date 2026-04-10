"""PMXT Archive orderbook parser for Polymarket backtesting.

Parses hourly orderbook snapshots from archive.pmxt.dev/Polymarket.
These are large Parquet files (442-730 MB) containing bid/ask levels
for all active markets at each hourly snapshot.

Usage:
    # Inspect schema of a downloaded file
    python data/pmxt_parser.py --inspect data/pmxt_orderbooks/polymarket_orderbook_2026-04-10T14.parquet

    # Parse orderbook into clean mid-price series
    python data/pmxt_parser.py --parse data/pmxt_orderbooks/*.parquet --output data/pmxt_prices.parquet

    # Download the most recent snapshot and inspect it
    python data/pmxt_parser.py --download-sample
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent
PMXT_DIR = DATA_DIR / "pmxt_orderbooks"
PMXT_ARCHIVE = "https://archive.pmxt.dev/Polymarket"
PMXT_R2_BASE = "https://r2.pmxt.dev"

# ────────────────────────────────────────────────────────────
# Schema discovery & inspection
# ────────────────────────────────────────────────────────────


def inspect_parquet_schema(filepath: Path) -> dict[str, Any]:
    """Read Parquet metadata without loading data. Returns column info."""
    pf = pq.ParquetFile(str(filepath))
    schema = pf.schema_arrow
    metadata = pf.metadata

    columns = []
    for i in range(len(schema)):
        field = schema.field(i)
        columns.append({
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        })

    info = {
        "filename": filepath.name,
        "size_mb": round(filepath.stat().st_size / (1024 * 1024), 1),
        "num_rows": metadata.num_rows,
        "num_columns": metadata.num_columns,
        "num_row_groups": metadata.num_row_groups,
        "columns": columns,
        "created_by": metadata.created_by,
    }

    # Read a tiny sample (first 100 rows) for value inspection
    sample = pf.read_row_group(0, columns=[c["name"] for c in columns]).to_pandas().head(100)
    for col_info in columns:
        name = col_info["name"]
        if name in sample.columns:
            series = sample[name]
            col_info["sample_values"] = series.dropna().head(3).tolist()
            col_info["nunique_sample"] = int(series.nunique())

    return info


def print_schema(info: dict[str, Any]) -> None:
    """Pretty-print schema inspection results."""
    print(f"\n{'='*60}")
    print(f"File: {info['filename']}")
    print(f"Size: {info['size_mb']} MB")
    print(f"Rows: {info['num_rows']:,}")
    print(f"Columns: {info['num_columns']}")
    print(f"Row groups: {info['num_row_groups']}")
    print(f"Created by: {info.get('created_by', 'unknown')}")
    print(f"{'='*60}")
    print(f"\n{'Column':<30} {'Type':<20} {'Samples'}")
    print(f"{'-'*30} {'-'*20} {'-'*40}")
    for col in info["columns"]:
        samples = col.get("sample_values", [])
        sample_str = str(samples)[:60] if samples else ""
        print(f"{col['name']:<30} {col['type']:<20} {sample_str}")
    print()


# ────────────────────────────────────────────────────────────
# Chunked Parquet reading for large files
# ────────────────────────────────────────────────────────────


def iter_row_groups(
    filepath: Path,
    columns: list[str] | None = None,
    batch_size: int = 1,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrames one row-group at a time to limit memory.

    PMXT files have multiple row groups; reading one at a time keeps
    peak memory well below the 442-730 MB file size.
    """
    pf = pq.ParquetFile(str(filepath))
    n_groups = pf.metadata.num_row_groups

    for start in range(0, n_groups, batch_size):
        end = min(start + batch_size, n_groups)
        tables = []
        for i in range(start, end):
            tbl = pf.read_row_group(i, columns=columns)
            tables.append(tbl)
        combined = pa.concat_tables(tables)
        yield combined.to_pandas()


# ────────────────────────────────────────────────────────────
# Orderbook parsing — extract bid/ask/mid/spread/depth
# ────────────────────────────────────────────────────────────


# Column name mappings: PMXT schema uses these names (discovered via inspection).
# If the schema differs, we detect columns dynamically.

# Expected columns (based on PMXT documentation and Polymarket CLOB):
#   - market / asset_id / token_id / condition_id  (market identifier)
#   - timestamp / snapshot_time                     (when the snapshot was taken)
#   - bids / asks or bid_price_X / ask_price_X      (orderbook levels)
#   - bid_size_X / ask_size_X                       (sizes at each level)
#   - outcome / outcome_tag                         (YES/NO)


def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Auto-detect column name mappings from whatever schema the file has.

    Returns a dict mapping logical names to actual column names.
    """
    cols = set(df.columns)
    mapping: dict[str, str] = {}

    # Market identifier
    for candidate in ["market", "asset_id", "token_id", "condition_id", "market_id"]:
        if candidate in cols:
            mapping["market_id"] = candidate
            break

    # Timestamp
    for candidate in ["timestamp", "snapshot_time", "time", "ts", "datetime", "date"]:
        if candidate in cols:
            mapping["timestamp"] = candidate
            break

    # Outcome (YES/NO)
    for candidate in ["outcome", "outcome_tag", "side", "token_outcome"]:
        if candidate in cols:
            mapping["outcome"] = candidate
            break

    # Best bid / ask prices
    for candidate in ["bid_price", "best_bid", "bids", "bid_price_0", "bid"]:
        if candidate in cols:
            mapping["best_bid"] = candidate
            break

    for candidate in ["ask_price", "best_ask", "asks", "ask_price_0", "ask"]:
        if candidate in cols:
            mapping["best_ask"] = candidate
            break

    # Bid/ask sizes
    for candidate in ["bid_size", "bid_qty", "bid_amount", "bid_size_0"]:
        if candidate in cols:
            mapping["bid_size"] = candidate
            break

    for candidate in ["ask_size", "ask_qty", "ask_amount", "ask_size_0"]:
        if candidate in cols:
            mapping["ask_size"] = candidate
            break

    # Price (pre-computed mid or last)
    for candidate in ["price", "mid_price", "last_price", "mid"]:
        if candidate in cols:
            mapping["price"] = candidate
            break

    # Question / slug
    for candidate in ["question", "title", "market_question", "description"]:
        if candidate in cols:
            mapping["question"] = candidate
            break

    for candidate in ["slug", "market_slug"]:
        if candidate in cols:
            mapping["slug"] = candidate
            break

    # Depth levels: collect all bid_price_N / ask_price_N / bid_size_N / ask_size_N
    bid_levels = sorted([c for c in cols if re.match(r"bid_price_\d+", c)])
    ask_levels = sorted([c for c in cols if re.match(r"ask_price_\d+", c)])
    bid_sizes = sorted([c for c in cols if re.match(r"bid_size_\d+", c)])
    ask_sizes = sorted([c for c in cols if re.match(r"ask_size_\d+", c)])

    if bid_levels:
        mapping["bid_levels"] = bid_levels  # type: ignore[assignment]
        mapping["ask_levels"] = ask_levels  # type: ignore[assignment]
        mapping["bid_sizes"] = bid_sizes    # type: ignore[assignment]
        mapping["ask_sizes"] = ask_sizes    # type: ignore[assignment]

    return mapping


def compute_orderbook_metrics(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Compute mid-price, spread, and book depth from raw orderbook data.

    Handles two orderbook formats:
    1. Level-based: bid_price_0..N, ask_price_0..N with corresponding sizes
    2. Summary: single best_bid, best_ask columns

    Returns DataFrame with columns:
        market_id, timestamp, outcome, mid_price, spread, spread_bps,
        best_bid, best_ask, bid_depth, ask_depth, total_depth
    """
    result = pd.DataFrame()

    # Market ID
    if "market_id" in col_map:
        result["market_id"] = df[col_map["market_id"]]

    # Timestamp
    if "timestamp" in col_map:
        ts_col = df[col_map["timestamp"]]
        # Try to parse if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(ts_col):
            result["timestamp"] = pd.to_datetime(ts_col, errors="coerce", utc=True)
        else:
            result["timestamp"] = ts_col
    else:
        result["timestamp"] = pd.NaT

    # Outcome
    if "outcome" in col_map:
        result["outcome"] = df[col_map["outcome"]]

    # Question / slug for identification
    if "question" in col_map:
        result["question"] = df[col_map["question"]]
    if "slug" in col_map:
        result["slug"] = df[col_map["slug"]]

    # --- Best bid / ask ---
    if "best_bid" in col_map and "best_ask" in col_map:
        best_bid = pd.to_numeric(df[col_map["best_bid"]], errors="coerce")
        best_ask = pd.to_numeric(df[col_map["best_ask"]], errors="coerce")
    elif "bid_levels" in col_map:
        # Use level 0 (top of book)
        bid_levels = col_map["bid_levels"]
        ask_levels = col_map["ask_levels"]
        best_bid = pd.to_numeric(df[bid_levels[0]], errors="coerce") if bid_levels else pd.Series(np.nan, index=df.index)
        best_ask = pd.to_numeric(df[ask_levels[0]], errors="coerce") if ask_levels else pd.Series(np.nan, index=df.index)
    elif "price" in col_map:
        # Fallback: use price as mid, estimate bid/ask with a small spread
        price = pd.to_numeric(df[col_map["price"]], errors="coerce")
        best_bid = price - 0.005
        best_ask = price + 0.005
    else:
        best_bid = pd.Series(np.nan, index=df.index)
        best_ask = pd.Series(np.nan, index=df.index)

    result["best_bid"] = best_bid
    result["best_ask"] = best_ask

    # Mid-price
    result["mid_price"] = (best_bid + best_ask) / 2.0

    # Spread
    result["spread"] = best_ask - best_bid
    result["spread_bps"] = (result["spread"] / result["mid_price"] * 10000).replace(
        [np.inf, -np.inf], np.nan
    )

    # --- Book depth (sum of sizes at all levels) ---
    bid_depth = pd.Series(0.0, index=df.index)
    ask_depth = pd.Series(0.0, index=df.index)

    if "bid_sizes" in col_map:
        for col_name in col_map["bid_sizes"]:
            bid_depth += pd.to_numeric(df[col_name], errors="coerce").fillna(0)
        for col_name in col_map["ask_sizes"]:
            ask_depth += pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    elif "bid_size" in col_map:
        bid_depth = pd.to_numeric(df[col_map["bid_size"]], errors="coerce").fillna(0)
        ask_depth = pd.to_numeric(df[col_map["ask_size"]], errors="coerce").fillna(0)

    result["bid_depth"] = bid_depth
    result["ask_depth"] = ask_depth
    result["total_depth"] = bid_depth + ask_depth

    return result


def parse_pmxt_file(
    filepath: Path,
    markets: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Parse a single PMXT Parquet file into clean orderbook metrics.

    Args:
        filepath: Path to Parquet file.
        markets: Optional list of market IDs to filter for.
        max_rows: Optional limit on rows to read (for testing).

    Returns:
        DataFrame with mid_price, spread, depth per market per snapshot.
    """
    logger.info("Parsing %s ...", filepath.name)

    # First pass: detect schema
    pf = pq.ParquetFile(str(filepath))
    sample = pf.read_row_group(0).to_pandas().head(10)
    col_map = _detect_columns(sample)

    logger.info("Detected column mapping: %s",
                {k: v for k, v in col_map.items()
                 if not isinstance(v, list)})

    # Determine which columns to actually read (minimize I/O)
    needed_cols = set()
    for key, val in col_map.items():
        if isinstance(val, list):
            needed_cols.update(val)
        elif isinstance(val, str):
            needed_cols.add(val)
    # Intersect with actual schema columns
    available = {f.name for f in pf.schema_arrow}
    read_cols = sorted(needed_cols & available)

    all_chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk_df in iter_row_groups(filepath, columns=read_cols):
        # Filter by market if requested
        if markets and "market_id" in col_map:
            mid_col = col_map["market_id"]
            chunk_df = chunk_df[chunk_df[mid_col].isin(markets)]

        if chunk_df.empty:
            continue

        metrics = compute_orderbook_metrics(chunk_df, col_map)
        all_chunks.append(metrics)
        total_rows += len(metrics)

        if max_rows and total_rows >= max_rows:
            break

    if not all_chunks:
        logger.warning("No data extracted from %s", filepath.name)
        return pd.DataFrame()

    result = pd.concat(all_chunks, ignore_index=True)

    # Drop rows where mid_price is NaN (no valid orderbook)
    before = len(result)
    result = result.dropna(subset=["mid_price"])
    if before - len(result) > 0:
        logger.info("Dropped %d rows with NaN mid_price", before - len(result))

    # Inject the snapshot hour from the filename
    ts_match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})", filepath.name)
    if ts_match:
        file_ts = datetime.strptime(
            f"{ts_match.group(1)}T{ts_match.group(2)}:00:00",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        result["snapshot_hour"] = file_ts

    logger.info(
        "Parsed %s: %d rows, %d unique markets",
        filepath.name,
        len(result),
        result["market_id"].nunique() if "market_id" in result.columns else 0,
    )
    return result


def parse_multiple_files(
    filepaths: list[Path],
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """Parse multiple PMXT files and concatenate into a time series.

    Sorts by (market_id, timestamp) for easy groupby operations.
    """
    all_dfs: list[pd.DataFrame] = []

    for fp in sorted(filepaths):
        df = parse_pmxt_file(fp, markets=markets)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort for time-series operations
    sort_cols = []
    if "market_id" in combined.columns:
        sort_cols.append("market_id")
    if "timestamp" in combined.columns:
        sort_cols.append("timestamp")
    elif "snapshot_hour" in combined.columns:
        sort_cols.append("snapshot_hour")
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    logger.info(
        "Combined %d files: %d total rows, %d unique markets",
        len(all_dfs),
        len(combined),
        combined["market_id"].nunique() if "market_id" in combined.columns else 0,
    )
    return combined


# ────────────────────────────────────────────────────────────
# Build price series from orderbook snapshots
# ────────────────────────────────────────────────────────────


def build_price_series(
    ob_df: pd.DataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """Aggregate orderbook snapshots into a regular price series.

    Groups by (market_id, outcome) and resamples to the requested frequency.
    Output columns: market_id, outcome, timestamp, mid_price, spread, depth.
    """
    if ob_df.empty:
        return pd.DataFrame()

    # Use snapshot_hour or timestamp for time axis
    time_col = "timestamp" if "timestamp" in ob_df.columns else "snapshot_hour"
    if time_col not in ob_df.columns:
        logger.warning("No time column found, cannot build price series")
        return ob_df

    group_cols = []
    if "market_id" in ob_df.columns:
        group_cols.append("market_id")
    if "outcome" in ob_df.columns:
        group_cols.append("outcome")

    if not group_cols:
        return ob_df

    results = []
    for keys, group in ob_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        group = group.set_index(time_col).sort_index()

        # Resample to requested frequency
        agg = group.resample(freq).agg({
            "mid_price": "mean",
            "spread": "mean",
            "best_bid": "last",
            "best_ask": "last",
            "bid_depth": "mean",
            "ask_depth": "mean",
            "total_depth": "mean",
        }).dropna(subset=["mid_price"])

        for i, col_name in enumerate(group_cols):
            agg[col_name] = keys[i]

        agg = agg.reset_index()
        results.append(agg)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ────────────────────────────────────────────────────────────
# Download helpers
# ────────────────────────────────────────────────────────────


def list_available_files(max_pages: int = 3) -> list[str]:
    """List available PMXT Parquet files from the archive."""
    files = []
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                f"{PMXT_ARCHIVE}?page={page}", timeout=30,
                headers={"User-Agent": "polymarket-backtester/1.0"},
            )
            resp.raise_for_status()
            matches = re.findall(
                r"(polymarket_orderbook_\d{4}-\d{2}-\d{2}T\d{2}\.parquet)",
                resp.text,
            )
            if not matches:
                break
            files.extend(matches)
        except Exception as e:
            logger.warning("PMXT listing error page %d: %s", page, e)
            break

    return sorted(set(files))


def download_sample_file() -> Path | None:
    """Download the most recent (and likely smallest) PMXT snapshot."""
    files = list_available_files(max_pages=1)
    if not files:
        logger.error("No PMXT files found on archive")
        return None

    # Pick the most recent
    filename = files[-1]
    PMXT_DIR.mkdir(parents=True, exist_ok=True)
    output = PMXT_DIR / filename

    if output.exists():
        logger.info("Already have %s (%.1f MB)", filename, output.stat().st_size / 1e6)
        return output

    url = f"{PMXT_R2_BASE}/{filename}"
    logger.info("Downloading %s from %s ...", filename, url)

    try:
        resp = requests.get(url, stream=True, timeout=300,
                            headers={"User-Agent": "polymarket-backtester/1.0"})
        resp.raise_for_status()

        downloaded = 0
        with open(output, "wb") as f:
            for chunk in resp.iter_content(chunk_size=128 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) == 0:
                    logger.info("  ... %.0f MB downloaded", downloaded / 1e6)

        logger.info("Downloaded %s (%.1f MB)", filename, downloaded / 1e6)
        return output

    except Exception as e:
        logger.error("Download failed: %s", e)
        if output.exists():
            output.unlink()
        return None


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PMXT orderbook parser")
    parser.add_argument("--inspect", type=str, help="Inspect a Parquet file schema")
    parser.add_argument("--parse", nargs="+", help="Parse Parquet file(s)")
    parser.add_argument("--output", type=str, help="Output Parquet file for parsed data")
    parser.add_argument("--download-sample", action="store_true", help="Download most recent snapshot")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--list-files", action="store_true", help="List available PMXT files")
    args = parser.parse_args()

    if args.list_files:
        files = list_available_files(max_pages=5)
        print(f"Available PMXT files ({len(files)}):")
        for f in files:
            print(f"  {f}")
        return

    if args.download_sample:
        path = download_sample_file()
        if path:
            info = inspect_parquet_schema(path)
            print_schema(info)
        return

    if args.inspect:
        path = Path(args.inspect)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        info = inspect_parquet_schema(path)
        print_schema(info)
        return

    if args.parse:
        filepaths = [Path(p) for p in args.parse]
        missing = [p for p in filepaths if not p.exists()]
        if missing:
            print(f"Files not found: {missing}")
            sys.exit(1)

        df = parse_multiple_files(filepaths)
        if df.empty:
            print("No data extracted")
            sys.exit(1)

        print(f"\nParsed data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample:\n{df.head(10)}")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(str(out_path), index=False)
            print(f"\nSaved to {out_path}")

        # Also build price series
        price_df = build_price_series(df)
        if not price_df.empty:
            print(f"\nPrice series shape: {price_df.shape}")
            print(f"Price series sample:\n{price_df.head(10)}")


if __name__ == "__main__":
    main()
