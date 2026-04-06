"""Entrypoint and async orchestrator for the weather arbitrage bot.

Coordinates all components with Telegram as the sole UI. No terminal
dashboard — everything is controlled and monitored through Telegram.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import re
import signal
import sys
from datetime import datetime, timezone
from typing import Any

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
from infra.telegram import TelegramUI
from infra.charts import (
    generate_pnl_chart,
    generate_forecast_chart,
    generate_positions_chart,
    generate_win_rate_chart,
)

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


class Bot:
    """Main orchestrator — Telegram is the sole UI."""

    def __init__(self, live_flag: bool = False) -> None:
        self.noaa = NOAAFeed()
        self.model = WeatherModel()
        self.polymarket = PolymarketFeed()
        self.falcon = FalconFeed()
        self.portfolio = Portfolio()
        self.risk = RiskManager(settings.STARTING_CAPITAL)
        self.executor = Executor(live_cli_flag=live_flag)
        self.db = Database()
        self.telegram = TelegramUI()
        self.signal_engine = SignalEngine(self.noaa, self.model, self.polymarket, self.falcon)
        self._running = False

    async def start(self) -> None:
        logger.info("Starting Weather Arb Bot in %s mode", self.executor.mode)

        # 1. Database
        await self.db.start()

        # 2. Register Telegram commands BEFORE starting (handlers must exist first)
        self._register_telegram_commands()

        # 3. Telegram
        await self.telegram.start()

        # 4. NOAA grid resolution + initial forecasts
        await self.noaa.start()
        self._register_noaa_callbacks()

        # 5. Polymarket contract discovery
        await self.polymarket.start()

        # 6. Register bucket definitions
        bucket_defs = self.polymarket.get_bucket_defs()
        self.model.register_buckets(bucket_defs)

        # 7. Falcon
        await self.falcon.start()

        # 8. Executor
        await self.executor.start()
        self.executor.on_trade(self._on_trade)

        # 9. Risk manager callbacks
        self.risk.on_kill_switch(self._on_kill_switch)
        self.risk.on_drawdown_warning(self._on_drawdown_warning)

        # 10. Signal engine callbacks
        self.signal_engine.on_signal(self._on_signal)

        self._running = True

        # Send startup message via Telegram
        await self.telegram.send_startup_message(
            mode=self.executor.mode,
            capital=settings.STARTING_CAPITAL,
            locations=list(LOCATIONS.keys()),
        )

        logger.info("Bot started in %s mode", self.executor.mode)

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
            asyncio.create_task(self.telegram.run_polling(), name="tg_polling"),
        ]

        if self.falcon.is_enabled:
            tasks.extend([
                asyncio.create_task(self.falcon.run_smart_money_loop(), name="falcon_sm"),
                asyncio.create_task(self.falcon.run_sentiment_loop(), name="falcon_sent"),
                asyncio.create_task(self.falcon.run_cross_market_loop(), name="falcon_cm"),
            ])

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False

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
                        self.telegram.log_signal(
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
        while self._running:
            await asyncio.sleep(settings.CONTRACT_SCAN_INTERVAL)
            try:
                bucket_defs = self.polymarket.get_bucket_defs()
                if bucket_defs:
                    self.model.register_buckets(bucket_defs)
            except Exception:
                logger.exception("Bucket refresh error")

    # ------------------------------------------------------------------
    # NOAA callbacks
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
            self.telegram.log_signal(
                f"SHIFT {evt.location} {evt.target_date}: "
                f"{evt.old_high_f:.0f} -> {evt.new_high_f:.0f}°F ({evt.shift_f:+.0f}°F)"
            )

        self.noaa.on_forecast(on_forecast)
        self.noaa.on_shift(on_shift)

    def _on_signal(self, sig: Signal) -> None:
        self.telegram.log_signal(
            f"{sig.location} {sig.bucket_label} "
            f"edge={sig.edge*100:.0f}% conf={sig.confidence:.2f} "
            f"ratio={sig.value_ratio:.1f}x -> {sig.signal_type}"
        )

    def _on_trade(self, result: TradeResult) -> None:
        m = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)", result.bucket_label)
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
    # Telegram command handlers (all async, return (text, image|None))
    # ------------------------------------------------------------------

    def _register_telegram_commands(self) -> None:
        self.telegram.register_command("start", self._cmd_start)
        self.telegram.register_command("status", self._cmd_status)
        self.telegram.register_command("positions", self._cmd_positions)
        self.telegram.register_command("trades", self._cmd_trades)
        self.telegram.register_command("pnl", self._cmd_pnl)
        self.telegram.register_command("weather", self._cmd_weather)
        self.telegram.register_command("forecast", self._cmd_forecast)
        self.telegram.register_command("signals", self._cmd_signals)
        self.telegram.register_command("settings", self._cmd_settings)
        self.telegram.register_command("kill", self._cmd_kill)
        self.telegram.register_command("resume", self._cmd_resume)
        self.telegram.register_command("help", self._cmd_help)
        self.telegram.register_callback("forecast", self._cb_forecast)

    async def _cmd_start(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        text = (
            "\U0001f321 <b>Weather Arb Bot</b>\n\n"
            f"Mode: <b>{self.executor.mode}</b>\n"
            f"Portfolio: ${stats['portfolio_value']:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Open: {stats['open_positions']} positions\n\n"
            "Tap a button below or use /help for all commands."
        )
        return text, None

    async def _cmd_status(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        kill = "\U0001f534 ACTIVE" if self.risk.kill_switch_active else "\U0001f7e2 off"
        falcon = "\U0001f7e2" if self.falcon.is_enabled else "\U0001f534"
        text = (
            "\U0001f4ca <b>STATUS</b>\n\n"
            f"Mode: <b>{self.executor.mode}</b>\n"
            f"Portfolio: <b>${stats['portfolio_value']:.2f}</b>\n"
            f"Cash: ${stats['cash']:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Drawdown: {stats['drawdown']*100:.1f}%\n\n"
            f"Open Positions: {stats['open_positions']}\n"
            f"Win Rate: {stats['win_rate']*100:.0f}% ({stats['total_trades']} trades)\n"
            f"Resolution: {stats['resolution_accuracy']*100:.0f}%\n\n"
            f"Kill Switch: {kill}\n"
            f"Falcon API: {falcon}\n"
            f"Uptime: {stats['uptime']}"
        )
        return text, None

    async def _cmd_positions(self) -> tuple[str, bytes | None]:
        positions = self.portfolio.get_open_positions()
        if not positions:
            return "No open positions.", None

        lines = ["\U0001f4cd <b>OPEN POSITIONS</b>\n"]
        chart_data = []
        for i, p in enumerate(positions, 1):
            edge_pct = ((p.current_price - p.entry_price) / max(p.entry_price, 0.001)) * 100
            pnl_emoji = "\U0001f7e2" if p.unrealized_pnl >= 0 else "\U0001f534"
            lines.append(
                f"{i}. {pnl_emoji} <b>{p.location}</b> {p.bucket_label}\n"
                f"   Entry: ${p.entry_price:.2f} | Now: ${p.current_price:.2f}\n"
                f"   Edge: {edge_pct:+.0f}% | uPnL: ${p.unrealized_pnl:+.2f}"
            )
            chart_data.append({
                "location": p.location,
                "bucket": p.bucket_label,
                "unrealized_pnl": p.unrealized_pnl,
            })

        text = "\n".join(lines)
        image = generate_positions_chart(chart_data)
        return text, image

    async def _cmd_trades(self) -> tuple[str, bytes | None]:
        trades = await self.db.get_recent_trades(10)
        if not trades:
            return "No trades yet.", None

        lines = ["\U0001f4dd <b>RECENT TRADES</b>\n"]
        for t in trades:
            side_emoji = "\U0001f7e2" if t["side"] == "BUY" else "\U0001f534"
            pnl_str = f"${t['pnl']:+.2f}" if t.get("pnl") is not None else "open"
            lines.append(
                f"{side_emoji} {t['location']} {t['bucket_label']} | "
                f"{t['side']} @ ${t['price']:.2f} | "
                f"${t['size_usd']:.2f} | {pnl_str}"
            )
        return "\n".join(lines), None

    async def _cmd_pnl(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        closed = self.portfolio._closed
        wins = sum(1 for c in closed if c.realized_pnl > 0)
        losses = len(closed) - wins
        total_pnl = sum(c.realized_pnl for c in closed)

        text = (
            "\U0001f4b0 <b>P&L BREAKDOWN</b>\n\n"
            f"Portfolio: <b>${stats['portfolio_value']:.2f}</b>\n"
            f"Starting: ${settings.STARTING_CAPITAL:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Realized: ${total_pnl:+.2f}\n"
            f"Wins: {wins} | Losses: {losses}"
        )

        image = generate_win_rate_chart(wins, losses, total_pnl)
        return text, image

    async def _cmd_weather(self) -> tuple[str, bytes | None]:
        lines = ["\U0001f326 <b>NOAA FORECASTS</b>\n"]
        today = datetime.now(timezone.utc).date()

        for loc_key in LOCATIONS:
            fc = self.noaa.get_latest(loc_key, today)
            if fc:
                agree = "\u2705" if fc.model_agreement else "\u274c"
                shift_str = f"{fc.forecast_shift_f:+.0f}°F" if fc.forecast_shift_f != 0 else "0°F"
                lines.append(
                    f"<b>{loc_key}</b>: High {fc.forecasted_high_f:.0f}°F "
                    f"(shift: {shift_str}) {agree}"
                )
                if fc.open_meteo_high_f:
                    lines.append(f"   Open-Meteo: {fc.open_meteo_high_f:.0f}°F")
            else:
                lines.append(f"<b>{loc_key}</b>: No forecast data")

        return "\n".join(lines), None

    async def _cmd_forecast(self) -> tuple[str, bytes | None]:
        """Show forecast distribution for the first city with data."""
        today = datetime.now(timezone.utc).date()
        for loc_key in LOCATIONS:
            fc = self.noaa.get_latest(loc_key, today)
            if not fc:
                continue
            probs = self.model.compute_probabilities(
                location=loc_key,
                target_date=today,
                forecast_high_f=fc.forecasted_high_f,
                lead_days=0,
                model_agreement=fc.model_agreement,
            )
            if not probs:
                continue

            # Build bucket data for chart
            bucket_defs = {b.contract_id: b for b in self.polymarket.get_bucket_defs()}
            chart_buckets = {}
            for cid, prob in probs.buckets.items():
                bdef = bucket_defs.get(cid)
                if bdef:
                    chart_buckets[bdef.label] = (bdef.temp_low_f, bdef.temp_high_f, prob)

            if chart_buckets:
                image = generate_forecast_chart(
                    loc_key, str(today), fc.forecasted_high_f, probs.sigma_f, chart_buckets
                )
                return f"\U0001f4ca Forecast distribution for <b>{loc_key}</b>", image

        return "No forecast data with active buckets available.", None

    async def _cb_forecast(self, data: str) -> tuple[str, bytes | None]:
        """Handle forecast:LOCATION callback."""
        parts = data.split(":")
        if len(parts) < 2:
            return "Invalid callback", None
        loc_key = parts[1]
        today = datetime.now(timezone.utc).date()
        fc = self.noaa.get_latest(loc_key, today)
        if not fc:
            return f"No forecast data for {loc_key}", None

        probs = self.model.compute_probabilities(
            location=loc_key,
            target_date=today,
            forecast_high_f=fc.forecasted_high_f,
            lead_days=0,
            model_agreement=fc.model_agreement,
        )
        if not probs:
            return f"No bucket probabilities for {loc_key}", None

        bucket_defs = {b.contract_id: b for b in self.polymarket.get_bucket_defs()}
        chart_buckets = {}
        for cid, prob in probs.buckets.items():
            bdef = bucket_defs.get(cid)
            if bdef:
                chart_buckets[bdef.label] = (bdef.temp_low_f, bdef.temp_high_f, prob)

        if chart_buckets:
            image = generate_forecast_chart(
                loc_key, str(today), fc.forecasted_high_f, probs.sigma_f, chart_buckets
            )
            return f"Forecast: <b>{loc_key}</b> — High {fc.forecasted_high_f:.0f}°F", image

        return f"No active buckets for {loc_key}", None

    async def _cmd_signals(self) -> tuple[str, bytes | None]:
        log = self.telegram.get_signal_log(15)
        if not log:
            return "No signals logged yet.", None
        lines = ["\U0001f50d <b>RECENT SIGNALS</b>\n"]
        for entry in log:
            lines.append(f"<code>{entry}</code>")
        return "\n".join(lines), None

    async def _cmd_settings(self) -> tuple[str, bytes | None]:
        text = (
            "\u2699\ufe0f <b>SETTINGS</b>\n\n"
            f"Entry Threshold: ${settings.ENTRY_THRESHOLD}\n"
            f"Exit Threshold: ${settings.EXIT_THRESHOLD}\n"
            f"Min Value Ratio: {settings.MIN_VALUE_RATIO}x\n"
            f"Min Edge: {settings.MIN_EDGE*100:.0f}%\n"
            f"Min Confidence: {settings.MIN_CONFIDENCE}\n"
            f"Kelly Fraction: {settings.KELLY_FRACTION}\n"
            f"Max Trade Size: ${settings.MAX_TRADE_SIZE}\n"
            f"Max Positions: {settings.MAX_OPEN_POSITIONS}\n"
            f"Max Drawdown: {settings.MAX_DAILY_DRAWDOWN*100:.0f}%\n"
            f"Scan Interval: {settings.SCAN_INTERVAL}s\n"
            f"Falcon Enabled: {self.falcon.is_enabled}\n"
            f"Live Trading: {self.executor.is_live}"
        )
        return text, None

    async def _cmd_kill(self) -> tuple[str, bytes | None]:
        self.risk.activate_kill_switch()
        return "\U0001f6a8 Kill switch activated. All trading halted.\nUse /resume to re-enable.", None

    async def _cmd_resume(self) -> tuple[str, bytes | None]:
        if self.risk.kill_switch_active:
            self.risk._kill_switch_active = False
            self.risk._drawdown_alerts_sent.clear()
            return "\U0001f7e2 Kill switch deactivated. Trading resumed.", None
        return "Kill switch was not active.", None

    async def _cmd_help(self) -> tuple[str, bytes | None]:
        text = (
            "\U0001f4d6 <b>COMMANDS</b>\n\n"
            "/start — Main menu\n"
            "/status — Portfolio status + stats\n"
            "/positions — Open positions + chart\n"
            "/trades — Recent trade history\n"
            "/pnl — P&L breakdown + win rate chart\n"
            "/weather — Current NOAA forecasts\n"
            "/forecast — Probability distribution chart\n"
            "/signals — Recent signal log\n"
            "/settings — Bot configuration\n"
            "/kill — Activate kill switch\n"
            "/resume — Deactivate kill switch\n"
            "/help — This message"
        )
        return text, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Weather Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    bot = Bot(live_flag=args.live)

    loop = asyncio.new_event_loop()

    def handle_signal(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(bot.shutdown()))

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
