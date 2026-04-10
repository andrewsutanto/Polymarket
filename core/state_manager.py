"""Persistent portfolio state manager with crash recovery and on-chain reconciliation.

Solves the P0 go-live blocker: portfolio state is no longer in-memory only.
Every trade, position change, and cash update is persisted synchronously to SQLite.
On startup, full state is reloaded from the database.  Periodic reconciliation
compares tracked USDC balance against the on-chain balance via Polygon RPC.
Graceful shutdown cancels open orders and flushes state.

Usage:
    state_mgr = StateManager(db_path="data/bot_state.db")
    state_mgr.install_signal_handlers(clob_client)

    # On startup:
    cash, positions, trades = state_mgr.load_state()

    # After every trade:
    state_mgr.persist_position(token_id, position_dict)
    state_mgr.persist_cash(cash)
    state_mgr.persist_trade(trade_dict)

    # Periodic reconciliation (call from scan loop):
    await state_mgr.reconcile_on_chain(expected_cash, wallet_address)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Polygon USDC contract on mainnet (chain 137)
POLYGON_USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
# Standard ERC-20 balanceOf ABI selector
BALANCE_OF_SELECTOR = "0x70a08231"
# Default Polygon RPC (free tier, sufficient for balance checks)
DEFAULT_POLYGON_RPC = "https://polygon-rpc.com"


class StateManager:
    """Synchronous SQLite-backed portfolio state persistence with crash recovery."""

    def __init__(self, db_path: str = "data/bot_state.db") -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = self._init_db()
        self._shutdown_callbacks: list[Callable[[], Any]] = []
        self._signal_handlers_installed = False
        self._shutting_down = False
        self._reconciliation_alerts: list[dict] = []
        self._polygon_rpc = os.getenv("POLYGON_RPC_URL", DEFAULT_POLYGON_RPC)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")  # WAL for crash safety
        conn.execute("PRAGMA synchronous=FULL")   # Full fsync on every commit
        conn.execute("PRAGMA busy_timeout=5000")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                token_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT,
                question TEXT,
                category TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                edge REAL,
                pnl REAL,
                mode TEXT,
                raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS reconciliation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tracked_cash REAL,
                on_chain_usdc REAL,
                discrepancy REAL,
                positions_count INTEGER,
                notes TEXT
            );
        """)
        return conn

    # ------------------------------------------------------------------
    # Crash recovery: load state from DB
    # ------------------------------------------------------------------

    def load_state(self) -> tuple[float | None, dict[str, dict], list[dict]]:
        """Reload full portfolio state from SQLite after a crash or restart.

        Returns:
            (cash_or_none, positions_dict, trades_list)
            cash is None if no prior state exists (fresh start).
        """
        # Load cash
        row = self._conn.execute(
            "SELECT value FROM portfolio_state WHERE key = 'cash'"
        ).fetchone()
        cash = float(row[0]) if row else None

        # Load starting_capital
        row = self._conn.execute(
            "SELECT value FROM portfolio_state WHERE key = 'starting_capital'"
        ).fetchone()
        starting_capital = float(row[0]) if row else None

        # Load mode
        row = self._conn.execute(
            "SELECT value FROM portfolio_state WHERE key = 'mode'"
        ).fetchone()
        mode = row[0] if row else None

        # Load positions
        positions: dict[str, dict] = {}
        for r in self._conn.execute("SELECT token_id, data FROM positions").fetchall():
            try:
                positions[r[0]] = json.loads(r[1])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt position data for %s, skipping", r[0])

        # Load trade count for session stats
        trades: list[dict] = []
        for r in self._conn.execute(
            "SELECT raw_json FROM trade_log ORDER BY id"
        ).fetchall():
            try:
                trades.append(json.loads(r[0]))
            except (json.JSONDecodeError, TypeError):
                pass

        if cash is not None:
            logger.info(
                "STATE RECOVERED: cash=$%.2f, %d positions, %d historical trades",
                cash, len(positions), len(trades),
            )
        else:
            logger.info("No prior state found — fresh start")

        meta = {
            "cash": cash,
            "starting_capital": starting_capital,
            "mode": mode,
        }
        return cash, positions, trades

    def load_metadata(self) -> dict[str, Any]:
        """Load all key-value metadata from portfolio_state table."""
        meta = {}
        for row in self._conn.execute("SELECT key, value FROM portfolio_state").fetchall():
            meta[row[0]] = row[1]
        return meta

    # ------------------------------------------------------------------
    # Synchronous persistence (called on every state change)
    # ------------------------------------------------------------------

    def persist_cash(self, cash: float) -> None:
        """Persist current cash balance. Called after every trade."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO portfolio_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("cash", str(cash), now),
        )
        # No explicit commit needed with isolation_level=None (autocommit)

    def persist_metadata(self, key: str, value: str) -> None:
        """Persist an arbitrary key-value pair (mode, starting_capital, etc.)."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO portfolio_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )

    def persist_position(self, token_id: str, position_data: dict) -> None:
        """Persist a single position. Called when a position is opened/updated."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO positions (token_id, data, updated_at) VALUES (?, ?, ?)",
            (token_id, json.dumps(position_data), now),
        )

    def remove_position(self, token_id: str) -> None:
        """Remove a closed position from persistence."""
        self._conn.execute("DELETE FROM positions WHERE token_id = ?", (token_id,))

    def persist_trade(self, trade: dict) -> None:
        """Persist a completed trade to the trade log. Synchronous write."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO trade_log (timestamp, market_id, question, category, direction, "
            "entry_price, exit_price, size_usd, edge, pnl, mode, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trade.get("timestamp", now),
                trade.get("market_id", ""),
                trade.get("question", ""),
                trade.get("category", ""),
                trade.get("direction", ""),
                trade.get("entry_price", 0.0),
                trade.get("exit_price", 0.0),
                trade.get("size_usd", 0.0),
                trade.get("edge", 0.0),
                trade.get("pnl", 0.0),
                trade.get("mode", "PAPER"),
                json.dumps(trade),
            ),
        )

    def persist_full_snapshot(self, cash: float, positions: dict[str, Any]) -> None:
        """Atomic full-state snapshot: cash + all positions in one transaction."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO portfolio_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("cash", str(cash), now),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO portfolio_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("last_snapshot", now, now),
            )
            # Clear and re-insert all positions atomically
            self._conn.execute("DELETE FROM positions")
            for token_id, pos_data in positions.items():
                if isinstance(pos_data, dict):
                    data_json = json.dumps(pos_data)
                else:
                    # Handle dataclass objects by converting to dict
                    data_json = json.dumps(pos_data.__dict__ if hasattr(pos_data, '__dict__') else str(pos_data))
                self._conn.execute(
                    "INSERT INTO positions (token_id, data, updated_at) VALUES (?, ?, ?)",
                    (token_id, data_json, now),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # On-chain reconciliation
    # ------------------------------------------------------------------

    async def reconcile_on_chain(
        self,
        tracked_cash: float,
        wallet_address: str | None = None,
        positions: dict | None = None,
        tolerance_usd: float = 1.0,
    ) -> dict:
        """Compare tracked USDC balance with on-chain balance via Polygon RPC.

        Args:
            tracked_cash: Our internally tracked cash balance.
            wallet_address: The wallet address to check. Reads from
                POLYMARKET_WALLET_ADDRESS env var if not provided.
            positions: Current positions dict for logging.
            tolerance_usd: Threshold below which discrepancies are ignored.

        Returns:
            Dict with reconciliation result including any discrepancy.
        """
        if not wallet_address:
            wallet_address = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
        if not wallet_address:
            # Try deriving from private key if available
            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if pk:
                try:
                    from eth_account import Account
                    wallet_address = Account.from_key(pk).address
                except ImportError:
                    pass

        if not wallet_address:
            return {"status": "skipped", "reason": "no_wallet_address"}

        try:
            on_chain_usdc = await self._fetch_usdc_balance(wallet_address)
        except Exception as exc:
            logger.warning("On-chain balance check failed: %s", exc)
            return {"status": "error", "reason": str(exc)}

        discrepancy = on_chain_usdc - tracked_cash
        now = datetime.now(timezone.utc).isoformat()
        positions_count = len(positions) if positions else 0

        result = {
            "status": "ok",
            "timestamp": now,
            "tracked_cash": tracked_cash,
            "on_chain_usdc": on_chain_usdc,
            "discrepancy": round(discrepancy, 4),
            "positions_count": positions_count,
        }

        # Log to DB
        notes = ""
        if abs(discrepancy) > tolerance_usd:
            notes = f"DISCREPANCY ALERT: ${discrepancy:+.4f}"
            result["status"] = "discrepancy"
            logger.warning(
                "BALANCE DISCREPANCY: tracked=$%.2f, on-chain=$%.4f, diff=$%.4f",
                tracked_cash, on_chain_usdc, discrepancy,
            )
            self._reconciliation_alerts.append(result)
        else:
            logger.info(
                "Reconciliation OK: tracked=$%.2f, on-chain=$%.4f",
                tracked_cash, on_chain_usdc,
            )

        self._conn.execute(
            "INSERT INTO reconciliation_log "
            "(timestamp, tracked_cash, on_chain_usdc, discrepancy, positions_count, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, tracked_cash, on_chain_usdc, round(discrepancy, 4), positions_count, notes),
        )

        return result

    async def _fetch_usdc_balance(self, wallet_address: str) -> float:
        """Fetch USDC balance from Polygon via eth_call JSON-RPC.

        Uses raw HTTP to avoid heavyweight web3 dependency.
        USDC on Polygon has 6 decimals.
        """
        import aiohttp

        # Encode balanceOf(address) call
        # Pad address to 32 bytes
        addr_clean = wallet_address.lower().replace("0x", "")
        call_data = BALANCE_OF_SELECTOR + "000000000000000000000000" + addr_clean

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": POLYGON_USDC_ADDRESS,
                    "data": call_data,
                },
                "latest",
            ],
            "id": 1,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._polygon_rpc,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()

        result_hex = data.get("result", "0x0")
        balance_raw = int(result_hex, 16)
        # USDC has 6 decimals
        return balance_raw / 1_000_000

    def get_reconciliation_alerts(self) -> list[dict]:
        """Return and clear pending reconciliation alerts."""
        alerts = list(self._reconciliation_alerts)
        self._reconciliation_alerts.clear()
        return alerts

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def on_shutdown(self, callback: Callable[[], Any]) -> None:
        """Register a callback to run on graceful shutdown."""
        self._shutdown_callbacks.append(callback)

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install SIGTERM/SIGINT handlers for graceful shutdown.

        On shutdown: runs all registered callbacks (e.g., cancel open orders),
        then persists final state and closes the DB connection.
        """
        if self._signal_handlers_installed:
            return

        def _handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.warning("Received %s — initiating graceful shutdown...", sig_name)
            if self._shutting_down:
                logger.warning("Double signal received, forcing exit")
                sys.exit(1)
            self._shutting_down = True

            # Run shutdown callbacks synchronously
            for cb in self._shutdown_callbacks:
                try:
                    result = cb()
                    # If callback returns a coroutine, schedule it
                    if asyncio.iscoroutine(result) and loop:
                        loop.create_task(result)
                except Exception:
                    logger.exception("Shutdown callback error")

            logger.info("State persisted, shutting down cleanly")
            self.close()
            sys.exit(0)

        # Install for both SIGTERM and SIGINT
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self._signal_handlers_installed = True
        logger.info("Graceful shutdown handlers installed (SIGINT, SIGTERM)")

    def install_async_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install async-aware signal handlers (preferred for asyncio apps).

        Uses loop.add_signal_handler on Unix, falls back to signal.signal on Windows.
        """
        if self._signal_handlers_installed:
            return

        async def _async_shutdown(sig_name: str) -> None:
            if self._shutting_down:
                return
            self._shutting_down = True
            logger.warning("Received %s — running async shutdown...", sig_name)

            for cb in self._shutdown_callbacks:
                try:
                    result = cb()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Async shutdown callback error")

            logger.info("Async shutdown complete, state persisted")
            self.close()

        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.ensure_future(
                        _async_shutdown(signal.Signals(s).name)
                    ),
                )
        else:
            # Windows: fall back to synchronous handler
            self.install_signal_handlers(loop)
            return

        self._signal_handlers_installed = True
        logger.info("Async graceful shutdown handlers installed")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass
