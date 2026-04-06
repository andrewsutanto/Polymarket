"""NOAA/NWS forecast polling and parsing.

Polls three NWS endpoints (hourly, daily, raw grid) on independent
schedules. Also optionally polls Open-Meteo for a second opinion.
Emits WeatherForecast dataclasses consumed by the probability model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

import aiohttp

from config import settings
from config.locations import LOCATIONS, CityConfig

logger = logging.getLogger(__name__)


@dataclass
class WeatherForecast:
    timestamp: datetime
    location: str
    target_date: date
    forecasted_high_f: float
    forecasted_low_f: float
    hourly_temps: list[float]
    model_run_time: datetime
    forecast_shift_f: float
    nws_confidence: str
    open_meteo_high_f: float | None
    model_agreement: bool
    update_count: int


@dataclass
class ForecastShiftEvent:
    location: str
    target_date: date
    old_high_f: float
    new_high_f: float
    shift_f: float
    model_run_time: datetime


class NOAAFeed:
    """Manages NWS + Open-Meteo forecast polling for all configured cities."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._grid_cache: dict[str, dict[str, Any]] = {}
        self._prev_forecasts: dict[tuple[str, date], float] = {}
        self._update_counts: dict[tuple[str, date], int] = {}
        self._last_update_times: dict[str, datetime] = {}
        self._latest: dict[tuple[str, date], WeatherForecast] = {}
        self._callbacks: list[Callable[[WeatherForecast], Any]] = []
        self._shift_callbacks: list[Callable[[ForecastShiftEvent], Any]] = []
        self._running = False

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": settings.NWS_USER_AGENT, "Accept": "application/geo+json"}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers=self.headers)
        await self._resolve_all_grids()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    def on_forecast(self, cb: Callable[[WeatherForecast], Any]) -> None:
        self._callbacks.append(cb)

    def on_shift(self, cb: Callable[[ForecastShiftEvent], Any]) -> None:
        self._shift_callbacks.append(cb)

    def get_latest(self, location: str, target_date: date) -> WeatherForecast | None:
        return self._latest.get((location, target_date))

    # ------------------------------------------------------------------
    # Grid resolution (cached permanently)
    # ------------------------------------------------------------------

    async def _resolve_all_grids(self) -> None:
        for loc_key, cfg in LOCATIONS.items():
            if loc_key not in self._grid_cache:
                grid = await self._resolve_grid(cfg)
                if grid:
                    self._grid_cache[loc_key] = grid
                    logger.info("Resolved grid for %s: %s", loc_key, grid)

    async def _resolve_grid(self, cfg: CityConfig) -> dict[str, Any] | None:
        url = f"{settings.NWS_API_BASE}/points/{cfg.lat},{cfg.lon}"
        data = await self._nws_get(url)
        if not data:
            return None
        props = data.get("properties", {})
        return {
            "office": props.get("gridId", ""),
            "gridX": props.get("gridX", 0),
            "gridY": props.get("gridY", 0),
            "forecast_url": props.get("forecast", ""),
            "forecast_hourly_url": props.get("forecastHourly", ""),
            "gridpoints_url": f"{settings.NWS_API_BASE}/gridpoints/{props.get('gridId')}/{props.get('gridX')},{props.get('gridY')}",
        }

    # ------------------------------------------------------------------
    # NWS request with exponential backoff
    # ------------------------------------------------------------------

    async def _nws_get(self, url: str, max_retries: int = 4) -> dict[str, Any] | None:
        assert self._session is not None
        delay = 2.0
        for attempt in range(max_retries):
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 503:
                        logger.warning("NWS 503 on %s (attempt %d)", url, attempt + 1)
                    else:
                        logger.warning("NWS HTTP %d on %s", resp.status, url)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("NWS request error on %s: %s", url, exc)
            await asyncio.sleep(delay + (attempt * 0.5))
            delay = min(delay * 2, 60.0)
        logger.error("NWS request failed after %d retries: %s", max_retries, url)
        return None

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def run_hourly_loop(self) -> None:
        while self._running:
            await self._poll_hourly_all()
            await asyncio.sleep(settings.NOAA_HOURLY_INTERVAL)

    async def run_daily_loop(self) -> None:
        while self._running:
            await self._poll_daily_all()
            await asyncio.sleep(settings.NOAA_DAILY_INTERVAL)

    async def run_grid_loop(self) -> None:
        while self._running:
            await self._poll_grid_all()
            await asyncio.sleep(settings.NOAA_GRID_INTERVAL)

    async def run_open_meteo_loop(self) -> None:
        while self._running:
            await self._poll_open_meteo_all()
            await asyncio.sleep(settings.OPEN_METEO_INTERVAL)

    # ------------------------------------------------------------------
    # Hourly forecast (primary signal)
    # ------------------------------------------------------------------

    async def _poll_hourly_all(self) -> None:
        for loc_key in LOCATIONS:
            grid = self._grid_cache.get(loc_key)
            if not grid:
                continue
            try:
                await self._poll_hourly(loc_key, grid)
            except Exception:
                logger.exception("Error polling hourly for %s", loc_key)
            await asyncio.sleep(1.0)

    async def _poll_hourly(self, loc_key: str, grid: dict[str, Any]) -> None:
        url = grid["forecast_hourly_url"]
        data = await self._nws_get(url)
        if not data:
            return

        props = data.get("properties", {})
        update_time_str = props.get("updateTime", "")
        periods = props.get("periods", [])
        if not periods:
            return

        update_time = _parse_iso(update_time_str) or datetime.now(timezone.utc)
        new_run = self._last_update_times.get(loc_key) != update_time
        if new_run:
            self._last_update_times[loc_key] = update_time

        by_date: dict[date, list[float]] = {}
        for p in periods:
            start = _parse_iso(p.get("startTime", ""))
            if not start:
                continue
            temp = _to_fahrenheit(p.get("temperature", 0), p.get("temperatureUnit", "F"))
            d = start.date()
            by_date.setdefault(d, []).append(temp)

        today = datetime.now(timezone.utc).date()
        for target, temps in by_date.items():
            lead = (target - today).days
            if lead < 0 or lead > settings.MAX_LEAD_DAYS:
                continue
            high = max(temps)
            low = min(temps)
            key = (loc_key, target)

            shift = 0.0
            prev = self._prev_forecasts.get(key)
            if prev is not None:
                shift = high - prev
            if new_run:
                self._prev_forecasts[key] = high
                self._update_counts[key] = self._update_counts.get(key, 0) + 1

            if abs(shift) >= settings.FORECAST_SHIFT_ALERT_F and new_run:
                evt = ForecastShiftEvent(
                    location=loc_key,
                    target_date=target,
                    old_high_f=prev or high,
                    new_high_f=high,
                    shift_f=shift,
                    model_run_time=update_time,
                )
                for cb in self._shift_callbacks:
                    try:
                        cb(evt)
                    except Exception:
                        logger.exception("Shift callback error")

            om_high = self._open_meteo_cache.get(key) if hasattr(self, "_open_meteo_cache") else None
            agreement = True
            if om_high is not None:
                agreement = abs(high - om_high) <= 2.0

            fc = WeatherForecast(
                timestamp=datetime.now(timezone.utc),
                location=loc_key,
                target_date=target,
                forecasted_high_f=high,
                forecasted_low_f=low,
                hourly_temps=temps,
                model_run_time=update_time,
                forecast_shift_f=shift,
                nws_confidence="",
                open_meteo_high_f=om_high,
                model_agreement=agreement,
                update_count=self._update_counts.get(key, 1),
            )
            self._latest[key] = fc
            for cb in self._callbacks:
                try:
                    cb(fc)
                except Exception:
                    logger.exception("Forecast callback error")

    # ------------------------------------------------------------------
    # Daily forecast (cross-reference)
    # ------------------------------------------------------------------

    async def _poll_daily_all(self) -> None:
        for loc_key in LOCATIONS:
            grid = self._grid_cache.get(loc_key)
            if not grid:
                continue
            try:
                await self._poll_daily(loc_key, grid)
            except Exception:
                logger.exception("Error polling daily for %s", loc_key)
            await asyncio.sleep(1.0)

    async def _poll_daily(self, loc_key: str, grid: dict[str, Any]) -> None:
        url = grid["forecast_url"]
        data = await self._nws_get(url)
        if not data:
            return
        props = data.get("properties", {})
        for period in props.get("periods", []):
            if period.get("isDaytime"):
                start = _parse_iso(period.get("startTime", ""))
                if not start:
                    continue
                target = start.date()
                key = (loc_key, target)
                existing = self._latest.get(key)
                if existing and not existing.nws_confidence:
                    detail = period.get("detailedForecast", "")
                    existing.nws_confidence = _extract_confidence(detail)

    # ------------------------------------------------------------------
    # Raw grid data (uncertainty estimation)
    # ------------------------------------------------------------------

    async def _poll_grid_all(self) -> None:
        for loc_key in LOCATIONS:
            grid = self._grid_cache.get(loc_key)
            if not grid:
                continue
            try:
                await self._poll_grid(loc_key, grid)
            except Exception:
                logger.exception("Error polling grid for %s", loc_key)
            await asyncio.sleep(1.0)

    async def _poll_grid(self, loc_key: str, grid: dict[str, Any]) -> None:
        url = grid["gridpoints_url"]
        data = await self._nws_get(url)
        if not data:
            return
        props = data.get("properties", {})
        max_temps = props.get("maxTemperature", {}).get("values", [])
        for entry in max_temps:
            valid_time = entry.get("validTime", "")
            if "/" in valid_time:
                valid_time = valid_time.split("/")[0]
            start = _parse_iso(valid_time)
            if not start:
                continue
            val = entry.get("value")
            if val is None:
                continue
            temp_f = val * 9.0 / 5.0 + 32.0
            target = start.date()
            key = (loc_key, target)
            existing = self._latest.get(key)
            if existing:
                existing.forecasted_high_f = round(
                    (existing.forecasted_high_f + temp_f) / 2.0, 1
                )

    # ------------------------------------------------------------------
    # Open-Meteo (second opinion)
    # ------------------------------------------------------------------

    _open_meteo_cache: dict[tuple[str, date], float] = {}

    async def _poll_open_meteo_all(self) -> None:
        assert self._session is not None
        for loc_key, cfg in LOCATIONS.items():
            try:
                await self._poll_open_meteo(loc_key, cfg)
            except Exception:
                logger.exception("Error polling Open-Meteo for %s", loc_key)
            await asyncio.sleep(0.5)

    async def _poll_open_meteo(self, loc_key: str, cfg: CityConfig) -> None:
        assert self._session is not None
        url = (
            f"{settings.OPEN_METEO_BASE}"
            f"?latitude={cfg.lat}&longitude={cfg.lon}"
            f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
            f"&timezone=auto&forecast_days={settings.MAX_LEAD_DAYS + 1}"
        )
        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        maxes = daily.get("temperature_2m_max", [])
        for d_str, t_max in zip(dates, maxes):
            if t_max is None:
                continue
            try:
                target = date.fromisoformat(d_str)
            except ValueError:
                continue
            key = (loc_key, target)
            self._open_meteo_cache[key] = float(t_max)
            existing = self._latest.get(key)
            if existing:
                existing.open_meteo_high_f = float(t_max)
                existing.model_agreement = abs(existing.forecasted_high_f - t_max) <= 2.0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_fahrenheit(temp: float, unit: str) -> float:
    if unit.upper().startswith("C"):
        return temp * 9.0 / 5.0 + 32.0
    return float(temp)


def _extract_confidence(detail: str) -> str:
    detail_lower = detail.lower()
    for keyword in ("slight chance", "chance", "likely", "certain"):
        if keyword in detail_lower:
            return keyword
    return ""
