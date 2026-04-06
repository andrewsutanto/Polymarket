"""Telegram alert service with rate limiting and command handlers.

Sends trade alerts, forecast shifts, drawdown warnings, and daily
summaries. Supports commands: /status, /kill, /trades, /pnl, /weather, /positions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from config import settings

logger = logging.getLogger(__name__)


class TelegramService:
    """Async Telegram bot for alerts and commands."""

    def __init__(self) -> None:
        self._bot: Any = None
        self._enabled = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        self._message_times: list[float] = []
        self._command_handlers: dict[str, Callable[[], str]] = {}
        self._running = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def register_command(self, command: str, handler: Callable[[], str]) -> None:
        self._command_handlers[command] = handler

    async def start(self) -> None:
        if not self._enabled:
            logger.warning("Telegram disabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
            return

        try:
            from telegram import Bot
            self._bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            self._running = True
            await self.send_message("Weather Arb Bot started")
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed — Telegram disabled")
            self._enabled = False
        except Exception as exc:
            logger.error("Telegram init failed: %s", exc)
            self._enabled = False

    async def stop(self) -> None:
        self._running = False
        if self._enabled:
            await self.send_message("Weather Arb Bot shutting down")

    async def send_message(self, text: str) -> bool:
        if not self._enabled or not self._bot:
            logger.info("[TG-disabled] %s", text[:100])
            return False

        if not self._rate_limit_ok():
            logger.debug("Telegram rate limited, dropping message")
            return False

        try:
            await self._bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
            )
            self._message_times.append(time.time())
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def _rate_limit_ok(self) -> bool:
        now = time.time()
        self._message_times = [t for t in self._message_times if now - t < 60]
        return len(self._message_times) < settings.TELEGRAM_RATE_LIMIT

    # ------------------------------------------------------------------
    # Alert formatters
    # ------------------------------------------------------------------

    async def alert_trade(
        self,
        mode: str,
        location: str,
        bucket: str,
        target_date: str,
        side: str,
        price: float,
        size: float,
        edge: float,
        confidence: float,
        forecast_high: float,
        kelly: float,
        value_ratio: float,
        portfolio_value: float,
        daily_pnl: float,
    ) -> None:
        text = (
            f"<b>Trade Executed [{mode}]</b>\n"
            f"{location} — {bucket} — {target_date}\n"
            f"{side} YES @ ${price:.2f} | Size: ${size:.2f}\n"
            f"Edge: {edge*100:.0f}% | Confidence: {confidence*100:.0f}%\n"
            f"NOAA forecast: High {forecast_high:.0f}°F\n"
            f"Kelly: {kelly:.2f} | Value ratio: {value_ratio:.1f}x\n"
            f"Portfolio: ${portfolio_value:.2f} | Daily P&L: ${daily_pnl:+.2f}"
        )
        await self.send_message(text)

    async def alert_forecast_shift(
        self,
        location: str,
        target_date: str,
        old_high: float,
        new_high: float,
        shift: float,
        model_run: str,
    ) -> None:
        direction = "+" if shift > 0 else ""
        text = (
            f"<b>FORECAST SHIFT</b>\n"
            f"{location} — {target_date}\n"
            f"High: {old_high:.0f}°F -> {new_high:.0f}°F ({direction}{shift:.0f}°F)\n"
            f"Model run: {model_run}"
        )
        await self.send_message(text)

    async def alert_drawdown(self, drawdown_pct: float) -> None:
        text = f"<b>DRAWDOWN WARNING: {drawdown_pct*100:.1f}%</b>"
        await self.send_message(text)

    async def alert_kill_switch(self) -> None:
        text = "<b>KILL SWITCH ACTIVATED</b>\nAll trading halted. Manual restart required."
        await self.send_message(text)

    async def alert_resolution(
        self, location: str, bucket: str, won: bool, pnl: float
    ) -> None:
        emoji = "YES" if won else "NO"
        text = (
            f"<b>Contract Resolved: {emoji}</b>\n"
            f"{location} — {bucket}\n"
            f"P&L: ${pnl:+.2f}"
        )
        await self.send_message(text)

    async def alert_connection(self, service: str, connected: bool) -> None:
        status = "connected" if connected else "disconnected"
        text = f"<b>Connection {status}: {service}</b>"
        await self.send_message(text)

    async def send_daily_summary(self, stats: dict[str, Any]) -> None:
        text = (
            f"<b>Daily Summary</b>\n"
            f"Portfolio: ${stats.get('portfolio_value', 0):.2f}\n"
            f"Daily P&L: ${stats.get('daily_pnl', 0):+.2f}\n"
            f"Win Rate: {stats.get('win_rate', 0)*100:.0f}%\n"
            f"Open Positions: {stats.get('open_positions', 0)}\n"
            f"Total Trades: {stats.get('total_trades', 0)}"
        )
        await self.send_message(text)

    # ------------------------------------------------------------------
    # Command polling (simple approach without webhook)
    # ------------------------------------------------------------------

    async def run_command_loop(self) -> None:
        """Poll for incoming Telegram commands."""
        if not self._enabled or not self._bot:
            return

        offset = 0
        while self._running:
            try:
                updates = await self._bot.get_updates(offset=offset, timeout=5)
                for update in updates:
                    offset = update.update_id + 1
                    if update.message and update.message.text:
                        await self._handle_command(update.message.text, update.message.chat_id)
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
            await asyncio.sleep(2)

    async def _handle_command(self, text: str, chat_id: int) -> None:
        if str(chat_id) != settings.TELEGRAM_CHAT_ID:
            return

        cmd = text.strip().split()[0].lower()
        handler = self._command_handlers.get(cmd)
        if handler:
            try:
                response = handler()
                await self.send_message(response)
            except Exception as exc:
                await self.send_message(f"Command error: {exc}")
