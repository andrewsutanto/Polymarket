"""Historical weather + odds data ingestion for backtesting.

Fetches NOAA archived forecasts, loads or synthesizes Polymarket odds,
and produces a unified BacktestDataset with aligned timestamps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.locations import LOCATIONS

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
NWS_API = "https://api.weather.gov"


@dataclass
class WeatherBar:
    """Single evaluation point with weather + odds data."""

    timestamp: datetime
    city: str
    forecast_high_f: float
    forecast_low_f: float
    forecast_shift_f: float
    model_prob: float
    market_price: float  # Polymarket implied prob
    book_depth: float
    spread: float
    volume: float
    lead_days: int
    is_forecast_update: bool
    data_source: str  # "real" or "synthetic"


@dataclass
class BacktestDataset:
    """Unified dataset for backtesting with aligned timestamps."""

    cities: list[str]
    bars: list[WeatherBar]
    start_date: datetime
    end_date: datetime
    n_bars: int
    odds_synthetic: bool

    def slice(self, start: datetime, end: datetime) -> "BacktestDataset":
        """Return subset within [start, end)."""
        filtered = [
            b for b in self.bars
            if start <= b.timestamp < end
        ]
        return BacktestDataset(
            cities=self.cities,
            bars=filtered,
            start_date=start,
            end_date=end,
            n_bars=len(filtered),
            odds_synthetic=self.odds_synthetic,
        )

    def get_city_bars(self, city: str) -> list[WeatherBar]:
        return [b for b in self.bars if b.city == city]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for b in self.bars:
            rows.append({
                "timestamp": b.timestamp,
                "city": b.city,
                "forecast_high_f": b.forecast_high_f,
                "forecast_low_f": b.forecast_low_f,
                "forecast_shift_f": b.forecast_shift_f,
                "model_prob": b.model_prob,
                "market_price": b.market_price,
                "book_depth": b.book_depth,
                "spread": b.spread,
                "volume": b.volume,
                "lead_days": b.lead_days,
                "is_forecast_update": b.is_forecast_update,
                "data_source": b.data_source,
            })
        return pd.DataFrame(rows)


def load_dataset(
    cities: list[str] | None = None,
    days: int = 180,
    eval_interval_min: int = 30,
    odds_noise_sigma: float = 0.02,
    odds_lag_range: tuple[int, int] = (1, 5),
    seed: int = 42,
    force_refresh: bool = False,
) -> BacktestDataset:
    """Load or generate a backtest dataset.

    Args:
        cities: List of city keys. Defaults to all configured.
        days: Number of historical days.
        eval_interval_min: Minutes between evaluation points.
        odds_noise_sigma: Noise for synthetic odds.
        odds_lag_range: Lag range for synthetic odds.
        seed: Random seed.
        force_refresh: Bypass cache.

    Returns:
        BacktestDataset with weather bars for all cities.
    """
    cities = cities or list(LOCATIONS.keys())
    rng = np.random.default_rng(seed)

    cache_key = f"weather_{'_'.join(sorted(cities))}_{days}d_{eval_interval_min}m"
    cache_path = CACHE_DIR / f"{cache_key}.parquet"

    if not force_refresh and cache_path.exists():
        logger.info("Loading cached dataset from %s", cache_path)
        df = pd.read_parquet(cache_path)
        return _df_to_dataset(df, cities)

    logger.info("Generating %d-day synthetic dataset for %s", days, cities)
    bars = _generate_synthetic_dataset(
        cities, days, eval_interval_min, odds_noise_sigma, odds_lag_range, rng
    )

    dataset = BacktestDataset(
        cities=cities,
        bars=bars,
        start_date=bars[0].timestamp if bars else datetime.now(timezone.utc),
        end_date=bars[-1].timestamp if bars else datetime.now(timezone.utc),
        n_bars=len(bars),
        odds_synthetic=True,
    )

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = dataset.to_dataframe()
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %d bars to %s", len(bars), cache_path)

    return dataset


def _generate_synthetic_dataset(
    cities: list[str],
    days: int,
    interval_min: int,
    noise_sigma: float,
    lag_range: tuple[int, int],
    rng: np.random.Generator,
) -> list[WeatherBar]:
    """Generate synthetic weather + odds bars.

    Simulates realistic temperature forecasts with daily cycles, random
    forecast updates, and lagged/noisy market prices.
    """
    bars: list[WeatherBar] = []
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    n_evals = int(days * 24 * 60 / interval_min)

    for city in cities:
        cfg = LOCATIONS.get(city)
        base_temp = _city_base_temp(city)

        # Generate forecast series
        temps = _generate_temperature_series(n_evals, base_temp, rng)
        shifts = np.concatenate([[0.0], np.diff(temps)])
        update_mask = _generate_update_schedule(n_evals, rng)

        # Model probabilities from temperature
        model_probs = _temps_to_probs(temps, base_temp)

        # Synthetic market prices with lag + noise
        market_prices = _add_lag_noise(model_probs, noise_sigma, lag_range, rng)

        for i in range(n_evals):
            ts = start + timedelta(minutes=i * interval_min)
            lead = max(1, (end.date() - ts.date()).days)

            bars.append(WeatherBar(
                timestamp=ts,
                city=city,
                forecast_high_f=float(temps[i]),
                forecast_low_f=float(temps[i] - 10 - rng.uniform(0, 5)),
                forecast_shift_f=float(shifts[i]),
                model_prob=float(model_probs[i]),
                market_price=float(market_prices[i]),
                book_depth=float(rng.uniform(20, 200)),
                spread=float(rng.uniform(0.01, 0.05)),
                volume=float(rng.uniform(50, 500)),
                lead_days=min(lead, 7),
                is_forecast_update=bool(update_mask[i]),
                data_source="synthetic",
            ))

    bars.sort(key=lambda b: b.timestamp)
    return bars


def _city_base_temp(city: str) -> float:
    """Approximate base temperature by city."""
    bases = {"NYC": 55, "Chicago": 50, "Seattle": 50, "Atlanta": 65, "Dallas": 70}
    return float(bases.get(city, 60))


def _generate_temperature_series(
    n: int, base: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate realistic temperature series with daily cycle and noise."""
    t = np.arange(n, dtype=float)
    daily_cycle = 10 * np.sin(2 * np.pi * t / (24 * 2))  # ~12hr cycle at 30min evals
    seasonal = 5 * np.sin(2 * np.pi * t / (24 * 2 * 365))
    noise = np.cumsum(rng.normal(0, 0.3, n))
    noise -= np.linspace(noise[0], noise[-1], n)  # Detrend
    return base + daily_cycle + seasonal + noise


def _generate_update_schedule(n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate NOAA update events (every ~12 evals on average)."""
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if rng.random() < 0.08:  # ~8% chance each eval
            mask[i] = True
    return mask


def _temps_to_probs(temps: np.ndarray, base: float) -> np.ndarray:
    """Convert temperatures to synthetic bucket probabilities."""
    from scipy.stats import norm
    sigma = 3.0
    bucket_center = base + 5  # Target bucket slightly above base
    probs = norm.cdf(temps, loc=bucket_center, scale=sigma)
    return np.clip(probs, 0.02, 0.98)


def _add_lag_noise(
    probs: np.ndarray,
    noise_sigma: float,
    lag_range: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add lag and noise to model probabilities to simulate market prices."""
    n = len(probs)
    lags = rng.integers(lag_range[0], lag_range[1] + 1, size=n)
    lagged = np.zeros(n)
    for i in range(n):
        src = max(0, i - lags[i])
        lagged[i] = probs[src]
    noise = rng.normal(0, noise_sigma, size=n)
    return np.clip(lagged + noise, 0.01, 0.99)


def _df_to_dataset(df: pd.DataFrame, cities: list[str]) -> BacktestDataset:
    """Reconstruct BacktestDataset from a cached DataFrame."""
    bars = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        bars.append(WeatherBar(
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            city=row["city"],
            forecast_high_f=row["forecast_high_f"],
            forecast_low_f=row["forecast_low_f"],
            forecast_shift_f=row["forecast_shift_f"],
            model_prob=row["model_prob"],
            market_price=row["market_price"],
            book_depth=row["book_depth"],
            spread=row["spread"],
            volume=row["volume"],
            lead_days=int(row["lead_days"]),
            is_forecast_update=bool(row["is_forecast_update"]),
            data_source=row.get("data_source", "synthetic"),
        ))
    bars.sort(key=lambda b: b.timestamp)
    synthetic = all(b.data_source == "synthetic" for b in bars)
    return BacktestDataset(
        cities=cities,
        bars=bars,
        start_date=bars[0].timestamp if bars else datetime.now(timezone.utc),
        end_date=bars[-1].timestamp if bars else datetime.now(timezone.utc),
        n_bars=len(bars),
        odds_synthetic=synthetic,
    )
