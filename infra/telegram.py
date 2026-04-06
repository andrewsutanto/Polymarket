"""Telegram bot as the sole UI for the weather arbitrage bot.

Uses python-telegram-bot's Application with handlers, inline keyboards,
callback queries, and chart image messages. Replaces the terminal dashboard.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any, Callable, Awaitable

from config import settings

logger = logging.getLogger(__name__)


class TelegramUI:
    """Full Telegram UI with commands, inline keyboards, and charts."""

    def __init__(self) -> None:
        self._app: Any = None
        self._bot: Any = None
        self._enabled = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        self._message_times: list[float] = []
        self._running = False
        # Command handlers are async callables returning (text, optional_image_bytes)
        self._command_handlers: dict[str, Callable[..., Awaitable[tuple[str, bytes | None]]]] = {}
        # Callback query handlers for inline keyboard buttons
        self._callback_handlers: dict[str, Callable[[str], Awaitable[tuple[str, bytes | None]]]] = {}

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def register_command(
        self, command: str, handler: Callable[..., Awaitable[tuple[str, bytes | None]]]
    ) -> None:
        """Register an async command handler. Returns (text, optional_image)."""
        self._command_handlers[command.lstrip("/")] = handler

    def register_callback(
        self, prefix: str, handler: Callable[[str], Awaitable[tuple[str, bytes | None]]]
    ) -> None:
        """Register a callback query handler for inline keyboard buttons."""
        self._callback_handlers[prefix] = handler

    async def start(self) -> None:
        if not self._enabled:
            logger.warning("Telegram disabled (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
            return

        try:
            from telegram import Update, BotCommand
            from telegram.ext import (
                Application, CommandHandler, CallbackQueryHandler, ContextTypes,
            )

            self._app = (
                Application.builder()
                .token(settings.TELEGRAM_BOT_TOKEN)
                .build()
            )
            self._bot = self._app.bot

            # Register all command handlers dynamically
            for cmd_name in list(self._command_handlers.keys()):
                self._app.add_handler(
                    CommandHandler(cmd_name, self._make_command_dispatcher(cmd_name))
                )

            # Register callback query handler for inline keyboards
            self._app.add_handler(CallbackQueryHandler(self._dispatch_callback))

            # Set bot commands menu
            commands = [
                BotCommand("start", "Show main menu"),
                BotCommand("status", "Portfolio status + stats"),
                BotCommand("positions", "Open positions with P&L chart"),
                BotCommand("trades", "Recent trade history"),
                BotCommand("pnl", "P&L chart and breakdown"),
                BotCommand("weather", "Current NOAA forecasts"),
                BotCommand("forecast", "Forecast distribution chart"),
                BotCommand("signals", "Recent signals log"),
                BotCommand("settings", "Bot settings and thresholds"),
                BotCommand("kill", "Activate kill switch"),
                BotCommand("resume", "Deactivate kill switch"),
                BotCommand("help", "Command reference"),
            ]
            await self._bot.set_my_commands(commands)

            self._running = True
            logger.info("Telegram UI initialized with %d commands", len(self._command_handlers))

        except ImportError:
            logger.error("python-telegram-bot not installed — Telegram disabled")
            self._enabled = False
        except Exception as exc:
            logger.error("Telegram init failed: %s", exc)
            self._enabled = False

    async def run_polling(self) -> None:
        """Start the Telegram polling loop (long-polling for updates)."""
        if not self._enabled or not self._app:
            return
        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram polling started")
            # Keep alive
            while self._running:
                await asyncio.sleep(1)
        except Exception as exc:
            logger.error("Telegram polling error: %s", exc)
        finally:
            if self._app.updater.running:
                await self._app.updater.stop()
            if self._app.running:
                await self._app.stop()
            await self._app.shutdown()

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    async def send_message(self, text: str, reply_markup: Any = None) -> bool:
        if not self._enabled or not self._bot:
            logger.info("[TG-disabled] %s", text[:100])
            return False
        if not self._rate_limit_ok():
            return False
        try:
            await self._bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
            self._message_times.append(time.time())
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    async def send_photo(self, image_bytes: bytes, caption: str = "") -> bool:
        if not self._enabled or not self._bot:
            return False
        if not self._rate_limit_ok():
            return False
        try:
            await self._bot.send_photo(
                chat_id=settings.TELEGRAM_CHAT_ID,
                photo=io.BytesIO(image_bytes),
                caption=caption,
                parse_mode="HTML",
            )
            self._message_times.append(time.time())
            return True
        except Exception as exc:
            logger.error("Telegram photo send failed: %s", exc)
            return False

    async def send_message_with_keyboard(
        self, text: str, buttons: list[list[tuple[str, str]]]
    ) -> bool:
        """Send a message with inline keyboard buttons.

        Args:
            text: Message text (HTML).
            buttons: 2D list of (label, callback_data) tuples.
        """
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [InlineKeyboardButton(label, callback_data=data) for label, data in row]
                for row in buttons
            ]
            markup = InlineKeyboardMarkup(keyboard)
            return await self.send_message(text, reply_markup=markup)
        except ImportError:
            return await self.send_message(text)

    # ------------------------------------------------------------------
    # Alert methods (called by the bot engine)
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
        emoji = "\U0001f321" if side == "BUY" else "\U0001f4b0"
        text = (
            f"{emoji} <b>TRADE [{mode}]</b>\n\n"
            f"<b>{location}</b> — {bucket} — {target_date}\n"
            f"{side} YES @ <b>${price:.2f}</b> | Size: ${size:.2f}\n\n"
            f"Edge: {edge*100:.0f}% | Confidence: {confidence*100:.0f}%\n"
            f"NOAA: High {forecast_high:.0f}°F\n"
            f"Kelly: {kelly:.2f} | Ratio: {value_ratio:.1f}x\n\n"
            f"\U0001f4bc ${portfolio_value:.2f} | P&L: ${daily_pnl:+.2f}"
        )
        buttons = [
            [("View Positions", "cmd:positions"), ("View P&L", "cmd:pnl")],
        ]
        await self.send_message_with_keyboard(text, buttons)

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
        emoji = "\U0001f525" if shift > 0 else "\u2744\ufe0f"
        text = (
            f"{emoji} <b>FORECAST SHIFT</b>\n\n"
            f"<b>{location}</b> — {target_date}\n"
            f"High: {old_high:.0f}°F \u2192 {new_high:.0f}°F ({direction}{shift:.0f}°F)\n"
            f"Model run: {model_run}"
        )
        buttons = [
            [("View Forecast", f"forecast:{location}"), ("View Positions", "cmd:positions")],
        ]
        await self.send_message_with_keyboard(text, buttons)

    async def alert_drawdown(self, drawdown_pct: float) -> None:
        text = (
            f"\u26a0\ufe0f <b>DRAWDOWN WARNING: {drawdown_pct*100:.1f}%</b>\n\n"
            f"Threshold: {settings.MAX_DAILY_DRAWDOWN*100:.0f}% (kill switch)"
        )
        buttons = [
            [("Kill Switch", "cmd:kill"), ("View Status", "cmd:status")],
        ]
        await self.send_message_with_keyboard(text, buttons)

    async def alert_kill_switch(self) -> None:
        text = (
            "\U0001f6a8 <b>KILL SWITCH ACTIVATED</b>\n\n"
            "All trading halted. Use /resume to re-enable."
        )
        await self.send_message(text)

    async def alert_resolution(
        self, location: str, bucket: str, won: bool, pnl: float
    ) -> None:
        emoji = "\u2705" if won else "\u274c"
        text = (
            f"{emoji} <b>RESOLVED: {'WON' if won else 'LOST'}</b>\n\n"
            f"{location} — {bucket}\n"
            f"P&L: <b>${pnl:+.2f}</b>"
        )
        buttons = [[("View Trades", "cmd:trades"), ("View P&L", "cmd:pnl")]]
        await self.send_message_with_keyboard(text, buttons)

    async def alert_connection(self, service: str, connected: bool) -> None:
        emoji = "\U0001f7e2" if connected else "\U0001f534"
        await self.send_message(f"{emoji} <b>{service}</b>: {'connected' if connected else 'disconnected'}")

    async def send_daily_summary(self, stats: dict[str, Any]) -> None:
        text = (
            "\U0001f4cb <b>DAILY SUMMARY</b>\n\n"
            f"Portfolio: <b>${stats.get('portfolio_value', 0):.2f}</b>\n"
            f"Daily P&L: ${stats.get('daily_pnl', 0):+.2f}\n"
            f"Win Rate: {stats.get('win_rate', 0)*100:.0f}%\n"
            f"Resolution Accuracy: {stats.get('resolution_accuracy', 0)*100:.0f}%\n"
            f"Open Positions: {stats.get('open_positions', 0)}\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Drawdown: {stats.get('drawdown', 0)*100:.1f}%\n"
            f"Uptime: {stats.get('uptime', '?')}"
        )
        await self.send_message(text)

    async def send_startup_message(self, mode: str, capital: float, locations: list[str]) -> None:
        text = (
            "\U0001f680 <b>Weather Arb Bot Started</b>\n\n"
            f"Mode: <b>{mode}</b>\n"
            f"Capital: ${capital:.2f}\n"
            f"Cities: {', '.join(locations)}\n\n"
            "Use /help for commands"
        )
        buttons = [
            [("Status", "cmd:status"), ("Weather", "cmd:weather")],
            [("Positions", "cmd:positions"), ("P&L", "cmd:pnl")],
        ]
        await self.send_message_with_keyboard(text, buttons)

    # ------------------------------------------------------------------
    # Signal log buffer (shown via /signals command)
    # ------------------------------------------------------------------

    _signal_log: list[str] = []

    def log_signal(self, line: str) -> None:
        self._signal_log.append(line)
        if len(self._signal_log) > 50:
            self._signal_log = self._signal_log[-50:]

    def get_signal_log(self, n: int = 15) -> list[str]:
        return self._signal_log[-n:]

    # ------------------------------------------------------------------
    # Dispatchers
    # ------------------------------------------------------------------

    def _make_command_dispatcher(self, cmd_name: str):
        """Create a handler function for a specific command."""
        async def handler(update: Any, context: Any) -> None:
            if not self._is_authorized(update):
                return
            callback = self._command_handlers.get(cmd_name)
            if not callback:
                return
            try:
                text, image = await callback()
                if image:
                    await update.message.reply_photo(
                        photo=io.BytesIO(image),
                        caption=text,
                        parse_mode="HTML",
                    )
                else:
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    # Add navigation keyboard to most responses
                    nav = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("Status", callback_data="cmd:status"),
                            InlineKeyboardButton("Positions", callback_data="cmd:positions"),
                            InlineKeyboardButton("P&L", callback_data="cmd:pnl"),
                        ],
                    ])
                    await update.message.reply_text(text, parse_mode="HTML", reply_markup=nav)
            except Exception as exc:
                logger.exception("Command handler error: %s", cmd_name)
                await update.message.reply_text(f"Error: {exc}")
        return handler

    async def _dispatch_callback(self, update: Any, context: Any) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        if not query:
            return
        await query.answer()

        data = query.data or ""

        # Handle "cmd:xxx" callbacks by running the command handler
        if data.startswith("cmd:"):
            cmd_name = data[4:]
            callback = self._command_handlers.get(cmd_name)
            if callback:
                try:
                    text, image = await callback()
                    if image:
                        await self._bot.send_photo(
                            chat_id=query.message.chat_id,
                            photo=io.BytesIO(image),
                            caption=text,
                            parse_mode="HTML",
                        )
                    else:
                        await query.edit_message_text(text, parse_mode="HTML")
                except Exception as exc:
                    logger.exception("Callback handler error: %s", data)
                    await query.edit_message_text(f"Error: {exc}")
            return

        # Handle prefixed callbacks
        prefix = data.split(":")[0]
        handler = self._callback_handlers.get(prefix)
        if handler:
            try:
                text, image = await handler(data)
                if image:
                    await self._bot.send_photo(
                        chat_id=query.message.chat_id,
                        photo=io.BytesIO(image),
                        caption=text,
                        parse_mode="HTML",
                    )
                else:
                    await query.edit_message_text(text, parse_mode="HTML")
            except Exception as exc:
                logger.exception("Callback handler error: %s", data)

    def _is_authorized(self, update: Any) -> bool:
        """Check if the message is from the authorized chat."""
        if not update.effective_chat:
            return False
        return str(update.effective_chat.id) == settings.TELEGRAM_CHAT_ID

    def _rate_limit_ok(self) -> bool:
        now = time.time()
        self._message_times = [t for t in self._message_times if now - t < 60]
        return len(self._message_times) < settings.TELEGRAM_RATE_LIMIT
