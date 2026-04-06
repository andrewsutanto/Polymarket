"""Entrypoint and async orchestrator for the weather arbitrage bot.

Coordinates all components: NOAA feeds, Polymarket polling, Falcon
enrichment, signal engine, risk manager, executor, portfolio tracker,
database, Telegram alerts, and the terminal dashboard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from config import settings
from config.locations import LOCATIONS
from core.noaa_feed import NOAAFeed, WeatherForecast, ForecastShiftEvent
from core.weather_model import WeatherModel
from core.polymarket_feed import PolymarketFeed
from core.falcon_intel import FalconFeed
from core.signal_engine import SignalEngine, Signal
from core.risk_manager import RiskManager
from core.executor import Executor, TradeResult
from core.portfolio import Portfolio
from infra.database import Database
from infra.telegram import TelegramService
from infra.dashboard import Dashboard

logger = logging.getLogger("weather_arb")


def setup_logging() -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.handlers.TimedRotatingFileHandler(
            os.path.join(settings.LOG_DIR, "bot.log"),
            when="midnight",
            backupCount=7,
        ),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


import logging.handlers


class Bot:
    """Main orchestrator that ties all components together."""

    def __init__(self, live_flag: bool = False, no_dashboard: bool = False) -> None:
        self.noaa = NOAAFeed()
        self.model = WeatherModel()
        self.polymarket = PolymarketFeed()
        self.falcon = FalconFeed()
        self.portfolio = Portfolio()
        self.risk = RiskManager(settings.STARTING_CAPITAL)
        self.executor = Executor(live_cli_flag=live_flag)
        self.db = Database()
        self.telegram = TelegramService()
        self.dashboard = Dashboard(mode=self.executor.mode)
        self.signal_engine = SignalEngine(self.noaa, self.model, self.polymarket, self.falcon)
        self._no_dashboard = no_dashboard
        self._running = False

    async def start(self) -> None:
        logger.info("Starting Weather Arb Bot in %s mode", self.executor.mode)

        # 1. Database
        await self.db.start()

        # 2. Telegram
        await self.telegram.start()
        self._register_telegram_commands()

        # 3. NOAA grid resolution + initial forecasts
        await self.noaa.start()
        self._register_noaa_callbacks()

        # 4. Polymarket contract discovery
        await self.polymarket.start()

        # 5. Register bucket definitions
        bucket_defs = self.polymarket.get_bucket_defs()
        self.model.register_buckets(bucket_defs)

        # 6. Falcon
        await self.falcon.start()

        # 7. Executor
        await self.executor.start()
        self.executor.on_trade(self._on_trade)

        # 8. Risk manager callbacks
        self.risk.on_kill_switch(self._on_kill_switch)
        self.risk.on_drawdown_warning(self._on_drawdown_warning)

        # 9. Signal engine callbacks
        self.signal_engine.on_signal(self._on_signal)

        self._running = True

        # Print mode banner
        if self.executor.is_live:
            print("\n" + "=" * 50)
            print("  === RUNNING IN LIVE MODE ===")
            print("=" * 50 + "\n")
        else:
            print("\n" + "=" * 50)
            print("  === RUNNING IN PAPER MODE ===")
            print("=" * 50 + "\n")

        await self.telegram.send_message(
            f"Bot started in {self.executor.mode} mode\n"
            f"Capital: ${settings.STARTING_CAPITAL}\n"
            f"Locations: {', '.join(LOCATIONS.keys())}"
        )

    async def run(self) -> None:
        await self.start()

        tasks = [
            asyncio.create_task(self.noaa.run_hourly_loop(), name="noaa_hourly"),
            asyncio.create_task(self.noaa.run_daily_loop(), name="noaa_daily"),
            asyncio.create_task(self.noaa.run_grid_loop(), name="noaa_grid"),
            asyncio.create_task(self.noaa.run_open_meteo_loop(), name="open_meteo"),
            asyncio.create_task(self.polymarket.run_poll_loop(), name="pm_poll"),
            asyncio.create_task(self.polymarket.run_discovery_loop(), name="pm_discover"),
            asyncio.create_task(self._scan_loop(), name="signal_scan"),
            asyncio.create_task(self._snapshot_loop(), name="snapshots"),
            asyncio.create_task(self._bucket_refresh_loop(), name="bucket_refresh"),
        ]

        if self.falcon.is_enabled:
            tasks.extend([
                asyncio.create_task(self.falcon.run_smart_money_loop(), name="falcon_sm"),
                asyncio.create_task(self.falcon.run_sentiment_loop(), name="falcon_sent"),
                asyncio.create_task(self.falcon.run_cross_market_loop(), name="falcon_cm"),
            ])

        if self.telegram.is_enabled:
            tasks.append(asyncio.create_task(self.telegram.run_command_loop(), name="tg_commands"))

        if not self._no_dashboard:
            tasks.append(asyncio.create_task(self.dashboard.run(), name="dashboard"))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False
        self.dashboard.stop()

        stats = self.portfolio.get_stats()
        await self.telegram.send_daily_summary(stats)

        await self.executor.stop()
        await self.noaa.stop()
        await self.polymarket.stop()
        await self.falcon.stop()
        await self.db.stop()
        await self.telegram.stop()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                self.risk.reset_cycle()
                self.signal_engine.set_open_positions(self.portfolio.get_open_position_map())
                self.risk.update_state(
                    available_cash=self.portfolio.cash,
                    open_positions=self.portfolio.get_position_sizes(),
                    session_pnl=self.portfolio.get_daily_pnl(),
                    portfolio_value=self.portfolio.get_portfolio_value(),
                )

                # Re-register buckets in case new contracts appeared
                bucket_defs = self.polymarket.get_bucket_defs()
                if bucket_defs:
                    self.model.register_buckets(bucket_defs)

                signals = self.signal_engine.scan()

                for sig in signals:
                    proposal = self.risk.evaluate(sig)
                    if proposal.approved:
                        result = await self.executor.execute(proposal)
                        if result:
                            self.portfolio.process_trade(result)
                    else:
                        self.dashboard.log_signal(
                            f"{sig.location} {sig.bucket_label} "
                            f"edge={sig.edge*100:.0f}% conf={sig.confidence:.2f} "
                            f"-> SKIP ({proposal.reject_reason})"
                        )

                # Mark-to-market
                prices = {
                    cid: snap.mid_price
                    for cid, snap in self.polymarket.get_all_snapshots().items()
                }
                self.portfolio.mark_to_market(prices)

                # Update dashboard
                self._refresh_dashboard()

            except Exception:
                logger.exception("Error in scan loop")

            await asyncio.sleep(settings.SCAN_INTERVAL)

    async def _snapshot_loop(self) -> None:
        while self._running:
            try:
                stats = self.portfolio.get_stats()
                await self.db.insert_snapshot(
                    portfolio_value=stats["portfolio_value"],
                    cash=stats["cash"],
                    open_positions=stats["open_positions"],
                    daily_pnl=stats["daily_pnl"],
                    drawdown=stats["drawdown"],
                )
            except Exception:
                logger.exception("Snapshot error")
            await asyncio.sleep(settings.SNAPSHOT_INTERVAL)

    async def _bucket_refresh_loop(self) -> None:
        """Periodically refresh bucket defs from newly discovered contracts."""
        while self._running:
            await asyncio.sleep(settings.CONTRACT_SCAN_INTERVAL)
            try:
                bucket_defs = self.polymarket.get_bucket_defs()
                if bucket_defs:
                    self.model.register_buckets(bucket_defs)
            except Exception:
                logger.exception("Bucket refresh error")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_noaa_callbacks(self) -> None:
        def on_forecast(fc: WeatherForecast) -> None:
            self.model.record_shift(fc.location, fc.target_date, fc.forecast_shift_f)
            asyncio.create_task(self.db.insert_forecast(
                timestamp=fc.timestamp.isoformat(),
                location=fc.location,
                target_date=str(fc.target_date),
                forecast_high_f=fc.forecasted_high_f,
                forecast_low_f=fc.forecasted_low_f,
                sigma_f=0.0,
                model_run_time=fc.model_run_time.isoformat(),
                source="NWS",
                model_agreement=fc.model_agreement,
            ))

        def on_shift(evt: ForecastShiftEvent) -> None:
            asyncio.create_task(self.telegram.alert_forecast_shift(
                location=evt.location,
                target_date=str(evt.target_date),
                old_high=evt.old_high_f,
                new_high=evt.new_high_f,
                shift=evt.shift_f,
                model_run=evt.model_run_time.strftime("%Hz"),
            ))
            self.dashboard.log_signal(
                f"SHIFT {evt.location} {evt.target_date}: "
                f"{evt.old_high_f:.0f} -> {evt.new_high_f:.0f}°F ({evt.shift_f:+.0f}°F)"
            )

        self.noaa.on_forecast(on_forecast)
        self.noaa.on_shift(on_shift)

    def _on_signal(self, sig: Signal) -> None:
        self.dashboard.log_signal(
            f"{sig.location} {sig.bucket_label} "
            f"edge={sig.edge*100:.0f}% conf={sig.confidence:.2f} "
            f"ratio={sig.value_ratio:.1f}x -> {sig.signal_type}"
        )

    def _on_trade(self, result: TradeResult) -> None:
        # Parse bucket boundaries from label
        import re
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", result.bucket_label)
        low_f = int(m.group(1)) if m else 0
        high_f = int(m.group(2)) if m else 0

        asyncio.create_task(self.db.insert_trade(
            timestamp=result.timestamp.isoformat(),
            contract_id=result.contract_id,
            location=result.location,
            target_date=result.target_date,
            bucket_label=result.bucket_label,
            bucket_low_f=low_f,
            bucket_high_f=high_f,
            side=result.side,
            size_usd=result.size_usd,
            price=result.price,
            edge=result.edge,
            confidence=result.confidence,
            model_prob=result.model_prob,
            market_price=result.market_price,
            forecast_high_f=result.forecast_high_f,
            kelly_fraction=result.kelly_fraction,
            mode=result.mode,
            status=result.status,
            notes=result.notes,
        ))

        stats = self.portfolio.get_stats()
        asyncio.create_task(self.telegram.alert_trade(
            mode=result.mode,
            location=result.location,
            bucket=result.bucket_label,
            target_date=result.target_date,
            side=result.side,
            price=result.price,
            size=result.size_usd,
            edge=result.edge,
            confidence=result.confidence,
            forecast_high=result.forecast_high_f,
            kelly=result.kelly_fraction,
            value_ratio=result.model_prob / max(result.market_price, 0.001),
            portfolio_value=stats["portfolio_value"],
            daily_pnl=stats["daily_pnl"],
        ))

    def _on_kill_switch(self) -> None:
        asyncio.create_task(self.telegram.alert_kill_switch())

    def _on_drawdown_warning(self, drawdown: float) -> None:
        asyncio.create_task(self.telegram.alert_drawdown(drawdown))

    # ------------------------------------------------------------------
    # Telegram commands
    # ------------------------------------------------------------------

    def _register_telegram_commands(self) -> None:
        self.telegram.register_command("/status", self._cmd_status)
        self.telegram.register_command("/kill", self._cmd_kill)
        self.telegram.register_command("/trades", self._cmd_trades)
        self.telegram.register_command("/pnl", self._cmd_pnl)
        self.telegram.register_command("/weather", self._cmd_weather)
        self.telegram.register_command("/positions", self._cmd_positions)

    def _cmd_status(self) -> str:
        stats = self.portfolio.get_stats()
        return (
            f"<b>Status</b>\n"
            f"Mode: {self.executor.mode}\n"
            f"Portfolio: ${stats['portfolio_value']:.2f}\n"
            f"Cash: ${stats['cash']:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Drawdown: {stats['drawdown']*100:.1f}%\n"
            f"Open: {stats['open_positions']} | Trades: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate']*100:.0f}%\n"
            f"Kill Switch: {'ACTIVE' if self.risk.kill_switch_active else 'off'}\n"
            f"Uptime: {stats['uptime']}"
        )

    def _cmd_kill(self) -> str:
        self.risk.activate_kill_switch()
        return "Kill switch activated. All trading halted."

    def _cmd_trades(self) -> str:
        closed = self.portfolio._closed[-10:]
        if not closed:
            return "No closed trades yet."
        lines = ["<b>Last 10 Trades</b>"]
        for c in reversed(closed):
            lines.append(
                f"{c.location} {c.bucket_label} | "
                f"${c.entry_price:.2f} -> ${c.exit_price:.2f} | "
                f"P&L: ${c.realized_pnl:+.2f} | {c.resolution}"
            )
        return "\n".join(lines)

    def _cmd_pnl(self) -> str:
        stats = self.portfolio.get_stats()
        return (
            f"<b>P&L</b>\n"
            f"Daily: ${stats['daily_pnl']:+.2f}\n"
            f"Portfolio: ${stats['portfolio_value']:.2f}\n"
            f"Starting: ${settings.STARTING_CAPITAL:.2f}"
        )

    def _cmd_weather(self) -> str:
        lines = ["<b>Current Forecasts</b>"]
        today = datetime.now(timezone.utc).date()
        for loc_key in LOCATIONS:
            fc = self.noaa.get_latest(loc_key, today)
            if fc:
                agree = "Y" if fc.model_agreement else "N"
                lines.append(
                    f"{loc_key}: High {fc.forecasted_high_f:.0f}°F "
                    f"(shift: {fc.forecast_shift_f:+.0f}°F, agree: {agree})"
                )
            else:
                lines.append(f"{loc_key}: No forecast data")
        return "\n".join(lines)

    def _cmd_positions(self) -> str:
        positions = self.portfolio.get_open_positions()
        if not positions:
            return "No open positions."
        lines = ["<b>Open Positions</b>"]
        for p in positions:
            edge = ((p.current_price - p.entry_price) / max(p.entry_price, 0.001)) * 100
            lines.append(
                f"{p.location} {p.bucket_label} | "
                f"Entry: ${p.entry_price:.2f} | Current: ${p.current_price:.2f} | "
                f"Edge: {edge:+.0f}% | uPnL: ${p.unrealized_pnl:+.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dashboard refresh
    # ------------------------------------------------------------------

    def _refresh_dashboard(self) -> None:
        self.dashboard.update_portfolio(self.portfolio.get_stats())

        # Forecasts
        today = datetime.now(timezone.utc).date()
        forecasts = []
        for loc_key in LOCATIONS:
            fc = self.noaa.get_latest(loc_key, today)
            if fc:
                forecasts.append({
                    "city": loc_key,
                    "date": str(fc.target_date),
                    "high": fc.forecasted_high_f,
                    "sigma": 0.0,
                    "shift": fc.forecast_shift_f,
                    "model_run": fc.model_run_time.strftime("%Hz") if fc.model_run_time else "",
                    "agreement": fc.model_agreement,
                })
        self.dashboard.update_forecasts(forecasts)

        # Positions
        positions = []
        for p in self.portfolio.get_open_positions():
            positions.append({
                "location": p.location,
                "bucket": p.bucket_label,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "size": p.qty * p.entry_price,
                "unrealized_pnl": p.unrealized_pnl,
            })
        self.dashboard.update_positions(positions)

        # Falcon status
        if self.falcon.is_enabled:
            self.dashboard.update_falcon(["Falcon API: connected"])
        else:
            self.dashboard.update_falcon(["Falcon API: disabled"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Weather Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires env + config flags too)")
    parser.add_argument("--no-dashboard", action="store_true", help="Run without terminal dashboard")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    bot = Bot(live_flag=args.live, no_dashboard=args.no_dashboard)

    loop = asyncio.new_event_loop()

    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(bot.shutdown()))

    import types
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    from typing import Any
    main()
