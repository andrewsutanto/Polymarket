"""SQLite database for trade logging, forecast tracking, and snapshots.

Uses aiosqlite for non-blocking writes. All operations are additive.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import aiosqlite

from config import settings

logger = logging.getLogger(__name__)

TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    contract_id     TEXT NOT NULL,
    location        TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    bucket_label    TEXT NOT NULL,
    bucket_low_f    INTEGER NOT NULL,
    bucket_high_f   INTEGER NOT NULL,
    side            TEXT NOT NULL,
    size_usd        REAL NOT NULL,
    price           REAL NOT NULL,
    edge            REAL NOT NULL,
    confidence      REAL NOT NULL,
    model_prob      REAL NOT NULL,
    market_price    REAL NOT NULL,
    forecast_high_f REAL NOT NULL,
    kelly_fraction  REAL NOT NULL,
    mode            TEXT NOT NULL,
    status          TEXT NOT NULL,
    pnl             REAL,
    closed_at       TEXT,
    resolution      TEXT,
    notes           TEXT
)
"""

FORECASTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    location        TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    forecast_high_f REAL NOT NULL,
    forecast_low_f  REAL NOT NULL,
    sigma_f         REAL NOT NULL,
    model_run_time  TEXT NOT NULL,
    source          TEXT NOT NULL,
    model_agreement BOOLEAN NOT NULL
)
"""

SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    portfolio_value REAL NOT NULL,
    cash            REAL NOT NULL,
    open_positions  INTEGER NOT NULL,
    daily_pnl       REAL NOT NULL,
    drawdown        REAL NOT NULL
)
"""


class Database:
    """Async SQLite database for persisting bot state."""

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or settings.DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def start(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(TRADES_SCHEMA)
        await self._db.execute(FORECASTS_SCHEMA)
        await self._db.execute(SNAPSHOTS_SCHEMA)
        await self._db.commit()
        logger.info("Database initialized at %s", self._path)

    async def stop(self) -> None:
        if self._db:
            await self._db.close()

    async def insert_trade(
        self,
        timestamp: str,
        contract_id: str,
        location: str,
        target_date: str,
        bucket_label: str,
        bucket_low_f: int,
        bucket_high_f: int,
        side: str,
        size_usd: float,
        price: float,
        edge: float,
        confidence: float,
        model_prob: float,
        market_price: float,
        forecast_high_f: float,
        kelly_fraction: float,
        mode: str,
        status: str,
        pnl: float | None = None,
        closed_at: str | None = None,
        resolution: str | None = None,
        notes: str | None = None,
    ) -> int:
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO trades (
                timestamp, contract_id, location, target_date, bucket_label,
                bucket_low_f, bucket_high_f, side, size_usd, price,
                edge, confidence, model_prob, market_price, forecast_high_f,
                kelly_fraction, mode, status, pnl, closed_at, resolution, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, contract_id, location, target_date, bucket_label,
                bucket_low_f, bucket_high_f, side, size_usd, price,
                edge, confidence, model_prob, market_price, forecast_high_f,
                kelly_fraction, mode, status, pnl, closed_at, resolution, notes,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or 0

    async def update_trade_resolution(
        self, contract_id: str, pnl: float, resolution: str
    ) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE trades SET pnl = ?, closed_at = ?, resolution = ?
               WHERE contract_id = ? AND resolution IS NULL""",
            (pnl, now, resolution, contract_id),
        )
        await self._db.commit()

    async def insert_forecast(
        self,
        timestamp: str,
        location: str,
        target_date: str,
        forecast_high_f: float,
        forecast_low_f: float,
        sigma_f: float,
        model_run_time: str,
        source: str,
        model_agreement: bool,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO forecasts (
                timestamp, location, target_date, forecast_high_f, forecast_low_f,
                sigma_f, model_run_time, source, model_agreement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, location, target_date, forecast_high_f, forecast_low_f,
                sigma_f, model_run_time, source, model_agreement,
            ),
        )
        await self._db.commit()

    async def insert_snapshot(
        self,
        portfolio_value: float,
        cash: float,
        open_positions: int,
        daily_pnl: float,
        drawdown: float,
    ) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO snapshots (
                timestamp, portfolio_value, cash, open_positions, daily_pnl, drawdown
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (now, portfolio_value, cash, open_positions, daily_pnl, drawdown),
        )
        await self._db.commit()

    async def get_recent_trades(self, limit: int = 10) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        columns = [desc[0] for desc in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    async def get_daily_pnl_breakdown(self) -> list[dict]:
        assert self._db is not None
        today = datetime.now(timezone.utc).date().isoformat()
        cursor = await self._db.execute(
            """SELECT location, bucket_label, side, price, size_usd, pnl, resolution
               FROM trades WHERE timestamp >= ? ORDER BY timestamp""",
            (today,),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
