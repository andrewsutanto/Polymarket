"""Entrypoint and async orchestrator for the Polymarket arbitrage scanner.

Scans the full Polymarket universe for mispricing opportunities across
all market categories. Telegram is the sole UI.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from config import settings
from core.gamma_feed import GammaFeed
from core.clob_feed import CLOBFeed
from core.market_scanner import MarketScanner, Opportunity
from core.risk_manager import RiskManager
from core.executor import Executor, TradeResult
from core.portfolio import Portfolio
from infra.database import Database
from infra.telegram import TelegramUI
from strategies import ALL_STRATEGIES
from strategies.ensemble import EnsembleStrategy
from strategies.base import Signal

logger = logging.getLogger("polymarket_scanner")


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
    """Main orchestrator — scans Polymarket universe via Telegram UI."""

    def __init__(self, live_flag: bool = False) -> None:
        self.gamma = GammaFeed()
        self.clob = CLOBFeed()
        self.portfolio = Portfolio()
        self.risk = RiskManager(settings.STARTING_CAPITAL)
        self.executor = Executor(live_cli_flag=live_flag)
        self.db = Database()
        self.telegram = TelegramUI()

        # Build all strategies
        strat_instances = [cls() for cls in ALL_STRATEGIES.values()]
        self.ensemble = EnsembleStrategy(strategies=strat_instances)
        self.scanner = MarketScanner(self.gamma, self.clob, strat_instances)

        self._running = False

    async def start(self) -> None:
        logger.info("Starting Polymarket Scanner in %s mode", self.executor.mode)

        await self.db.start()
        self._register_telegram_commands()
        await self.telegram.start()
        await self.gamma.start()
        await self.clob.start()
        await self.scanner.start()
        await self.executor.start()
        self.executor.on_trade(self._on_trade)
        self.risk.on_kill_switch(self._on_kill_switch)
        self.risk.on_drawdown_warning(self._on_drawdown_warning)
        self.scanner.on_signal(self._on_signal)

        self._running = True

        await self.telegram.send_startup_message(
            mode=self.executor.mode,
            capital=settings.STARTING_CAPITAL,
            locations=["All Polymarket Markets"],
        )

        logger.info(
            "Bot started — %d markets discovered, %d strategies active",
            self.gamma.market_count, len(ALL_STRATEGIES),
        )

    async def run(self) -> None:
        await self.start()

        tasks = [
            asyncio.create_task(self.gamma.run_poll_loop(), name="gamma_poll"),
            asyncio.create_task(self.clob.run_poll_loop(), name="clob_poll"),
            asyncio.create_task(self.scanner.run_full_scan_loop(), name="full_scan"),
            asyncio.create_task(self._signal_scan_loop(), name="signal_scan"),
            asyncio.create_task(self._snapshot_loop(), name="snapshots"),
            asyncio.create_task(self.telegram.run_polling(), name="tg_polling"),
        ]

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
        await self.gamma.stop()
        await self.clob.stop()
        await self.scanner.stop()
        await self.db.stop()
        await self.telegram.stop()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    async def _signal_scan_loop(self) -> None:
        """Run strategies on tracked markets every SCAN_INTERVAL."""
        while self._running:
            try:
                self.risk.reset_cycle()
                self.risk.update_state(
                    available_cash=self.portfolio.cash,
                    open_positions=self.portfolio.get_position_sizes(),
                    session_pnl=self.portfolio.get_daily_pnl(),
                    portfolio_value=self.portfolio.get_portfolio_value(),
                )

                if self.risk.kill_switch_active:
                    await asyncio.sleep(settings.SCAN_INTERVAL)
                    continue

                signals = await self.scanner.run_signal_scan()

                for sig in signals:
                    proposal = self.risk.evaluate_signal(sig)
                    if proposal:
                        result = await self.executor.execute(proposal)
                        if result:
                            self.portfolio.process_trade(result)
                    else:
                        self.telegram.log_signal(
                            f"{sig.category}/{sig.market_slug[:20]} "
                            f"{sig.strategy_name} edge={sig.edge:.1%} -> SKIP"
                        )

                # Mark-to-market
                prices = {}
                for tid, snap in self.clob.get_all_snapshots().items():
                    prices[tid] = snap.mid_price
                self.portfolio.mark_to_market(prices)

            except Exception:
                logger.exception("Error in signal scan loop")

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

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_signal(self, sig: Signal) -> None:
        self.telegram.log_signal(
            f"{sig.category}/{sig.market_slug[:20]} "
            f"{sig.strategy_name} {sig.direction} edge={sig.edge:.1%} str={sig.strength:.2f}"
        )

    def _on_trade(self, result: TradeResult) -> None:
        asyncio.create_task(self.db.insert_trade(
            timestamp=result.timestamp.isoformat(),
            contract_id=result.contract_id,
            location=result.location,
            target_date=result.target_date,
            bucket_label=result.bucket_label,
            bucket_low_f=0,
            bucket_high_f=0,
            side=result.side,
            size_usd=result.size_usd,
            price=result.price,
            edge=result.edge,
            confidence=result.confidence,
            model_prob=result.model_prob,
            market_price=result.market_price,
            forecast_high_f=0,
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
            forecast_high=0,
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
        self.telegram.register_command("start", self._cmd_start)
        self.telegram.register_command("status", self._cmd_status)
        self.telegram.register_command("scan", self._cmd_scan)
        self.telegram.register_command("markets", self._cmd_markets)
        self.telegram.register_command("opportunities", self._cmd_opportunities)
        self.telegram.register_command("positions", self._cmd_positions)
        self.telegram.register_command("trades", self._cmd_trades)
        self.telegram.register_command("pnl", self._cmd_pnl)
        self.telegram.register_command("signals", self._cmd_signals)
        self.telegram.register_command("settings", self._cmd_settings)
        self.telegram.register_command("kill", self._cmd_kill)
        self.telegram.register_command("resume", self._cmd_resume)
        self.telegram.register_command("help", self._cmd_help)

    async def _cmd_start(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        opps = len(self.scanner.opportunities)
        tracked = self.scanner.tracked_count
        text = (
            "\U0001f50d <b>Polymarket Scanner</b>\n\n"
            f"Mode: <b>{self.executor.mode}</b>\n"
            f"Portfolio: ${stats['portfolio_value']:.2f}\n"
            f"Markets discovered: {self.gamma.market_count}\n"
            f"Opportunities: {opps}\n"
            f"Tracking (CLOB): {tracked}\n"
            f"Strategies: {len(ALL_STRATEGIES)}\n\n"
            "Use /scan to run a full scan or /help for commands."
        )
        return text, None

    async def _cmd_status(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        kill = "\U0001f534 ACTIVE" if self.risk.kill_switch_active else "\U0001f7e2 off"
        text = (
            "\U0001f4ca <b>STATUS</b>\n\n"
            f"Mode: <b>{self.executor.mode}</b>\n"
            f"Portfolio: <b>${stats['portfolio_value']:.2f}</b>\n"
            f"Cash: ${stats['cash']:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Drawdown: {stats['drawdown']*100:.1f}%\n\n"
            f"Markets: {self.gamma.market_count}\n"
            f"Tracked: {self.scanner.tracked_count}\n"
            f"Opportunities: {len(self.scanner.opportunities)}\n"
            f"Relationships: {len(self.scanner.relationships)}\n\n"
            f"Open Positions: {stats['open_positions']}\n"
            f"Win Rate: {stats['win_rate']*100:.0f}% ({stats['total_trades']} trades)\n"
            f"Kill Switch: {kill}\n"
            f"Uptime: {stats['uptime']}"
        )
        return text, None

    async def _cmd_scan(self) -> tuple[str, bytes | None]:
        opps = await self.scanner.full_scan()
        if not opps:
            return "No opportunities found.", None

        lines = [f"\U0001f50d <b>SCAN RESULTS</b> ({len(opps)} found)\n"]
        for i, o in enumerate(opps[:10], 1):
            emoji = "\U0001f7e2" if o.direction == "BUY" else "\U0001f534"
            lines.append(
                f"{i}. {emoji} <b>{o.category}</b> | {o.strategy_name}\n"
                f"   {o.question[:60]}\n"
                f"   {o.direction} {o.outcome} | Edge: {o.edge:.1%} | "
                f"Vol: ${o.volume_24h:,.0f} | Liq: ${o.liquidity:,.0f}"
            )
        return "\n".join(lines), None

    async def _cmd_markets(self) -> tuple[str, bytes | None]:
        markets = self.gamma.markets
        categories: dict[str, int] = {}
        for m in markets.values():
            categories[m.category] = categories.get(m.category, 0) + 1

        lines = [f"\U0001f4ca <b>MARKETS</b> ({len(markets)} total)\n"]
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")

        top = self.gamma.get_top_by_volume(5)
        if top:
            lines.append("\n<b>Top by Volume:</b>")
            for m in top:
                lines.append(
                    f"  ${m.volume_24h:,.0f} | {m.question[:50]}"
                )

        return "\n".join(lines), None

    async def _cmd_opportunities(self) -> tuple[str, bytes | None]:
        opps = self.scanner.opportunities
        if not opps:
            return "No opportunities. Run /scan first.", None

        lines = [f"\U0001f4b0 <b>TOP OPPORTUNITIES</b> ({len(opps)} total)\n"]
        for i, o in enumerate(opps[:15], 1):
            emoji = "\U0001f7e2" if o.direction == "BUY" else "\U0001f534"
            lines.append(
                f"{i}. {emoji} {o.strategy_name} | {o.category}\n"
                f"   {o.question[:55]}\n"
                f"   {o.direction} @ ${o.yes_price:.2f} | Edge: {o.edge:.1%}"
            )
        return "\n".join(lines), None

    async def _cmd_positions(self) -> tuple[str, bytes | None]:
        positions = self.portfolio.get_open_positions()
        if not positions:
            return "No open positions.", None
        lines = ["\U0001f4cd <b>OPEN POSITIONS</b>\n"]
        for i, p in enumerate(positions, 1):
            pnl_emoji = "\U0001f7e2" if p.unrealized_pnl >= 0 else "\U0001f534"
            lines.append(
                f"{i}. {pnl_emoji} {p.location} {p.bucket_label}\n"
                f"   Entry: ${p.entry_price:.3f} | Now: ${p.current_price:.3f} | "
                f"uPnL: ${p.unrealized_pnl:+.4f}"
            )
        return "\n".join(lines), None

    async def _cmd_trades(self) -> tuple[str, bytes | None]:
        trades = await self.db.get_recent_trades(10)
        if not trades:
            return "No trades yet.", None
        lines = ["\U0001f4dd <b>RECENT TRADES</b>\n"]
        for t in trades:
            pnl = f"${t['pnl']:+.4f}" if t.get("pnl") is not None else "open"
            lines.append(
                f"{t['side']} {t['location'][:20]} @ ${t['price']:.3f} | "
                f"${t['size_usd']:.2f} | {pnl}"
            )
        return "\n".join(lines), None

    async def _cmd_pnl(self) -> tuple[str, bytes | None]:
        stats = self.portfolio.get_stats()
        text = (
            "\U0001f4b0 <b>P&L</b>\n\n"
            f"Portfolio: <b>${stats['portfolio_value']:.2f}</b>\n"
            f"Starting: ${settings.STARTING_CAPITAL:.2f}\n"
            f"Daily P&L: ${stats['daily_pnl']:+.2f}\n"
            f"Win Rate: {stats['win_rate']*100:.0f}%\n"
            f"Total Trades: {stats['total_trades']}"
        )
        return text, None

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
            f"Min Edge: {settings.MIN_EDGE:.0%}\n"
            f"Min Confidence: {settings.MIN_CONFIDENCE}\n"
            f"Kelly Fraction: {settings.KELLY_FRACTION}\n"
            f"Max Trade: ${settings.MAX_TRADE_SIZE}\n"
            f"Max Positions: {settings.MAX_OPEN_POSITIONS}\n"
            f"Max Drawdown: {settings.MAX_DAILY_DRAWDOWN:.0%}\n"
            f"Min Liquidity: ${settings.MIN_MARKET_LIQUIDITY}\n"
            f"Min Volume 24h: ${settings.MIN_MARKET_VOLUME_24H}\n"
            f"Strategies: {len(ALL_STRATEGIES)}\n"
            f"Tracked Markets: {settings.CLOB_TOP_N_TRACKED}"
        )
        return text, None

    async def _cmd_kill(self) -> tuple[str, bytes | None]:
        self.risk.activate_kill_switch()
        return "\U0001f6a8 Kill switch activated. Use /resume to re-enable.", None

    async def _cmd_resume(self) -> tuple[str, bytes | None]:
        if self.risk.kill_switch_active:
            self.risk._kill_switch_active = False
            self.risk._drawdown_alerts_sent.clear()
            return "\U0001f7e2 Kill switch deactivated. Trading resumed.", None
        return "Kill switch was not active.", None

    async def _cmd_help(self) -> tuple[str, bytes | None]:
        text = (
            "\U0001f4d6 <b>COMMANDS</b>\n\n"
            "/start — Dashboard overview\n"
            "/status — Portfolio + system status\n"
            "/scan — Run full market scan now\n"
            "/markets — Market universe breakdown\n"
            "/opportunities — Top ranked opportunities\n"
            "/positions — Open positions\n"
            "/trades — Recent trade history\n"
            "/pnl — P&L breakdown\n"
            "/signals — Signal log\n"
            "/settings — Configuration\n"
            "/kill — Activate kill switch\n"
            "/resume — Deactivate kill switch\n"
            "/help — This message"
        )
        return text, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Scanner")
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
