"""Terminal dashboard using the rich library.

Displays portfolio state, NOAA forecasts, open positions, recent
trades, signal log, and Falcon intel in a live-updating layout.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import settings

logger = logging.getLogger(__name__)


class Dashboard:
    """Rich terminal dashboard for the weather arb bot."""

    def __init__(self, mode: str = "PAPER") -> None:
        self._console = Console()
        self._mode = mode
        self._running = False
        self._portfolio_stats: dict[str, Any] = {}
        self._forecasts: list[dict[str, Any]] = []
        self._positions: list[dict[str, Any]] = []
        self._trades: list[dict[str, Any]] = []
        self._signals: list[str] = []
        self._falcon_lines: list[str] = []

    def update_portfolio(self, stats: dict[str, Any]) -> None:
        self._portfolio_stats = stats

    def update_forecasts(self, forecasts: list[dict[str, Any]]) -> None:
        self._forecasts = forecasts

    def update_positions(self, positions: list[dict[str, Any]]) -> None:
        self._positions = positions

    def update_trades(self, trades: list[dict[str, Any]]) -> None:
        self._trades = trades[-10:]

    def log_signal(self, line: str) -> None:
        self._signals.append(line)
        if len(self._signals) > 20:
            self._signals = self._signals[-20:]

    def update_falcon(self, lines: list[str]) -> None:
        self._falcon_lines = lines

    async def run(self) -> None:
        self._running = True
        refresh_sec = settings.DASHBOARD_REFRESH_MS / 1000.0

        with Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=1000.0 / settings.DASHBOARD_REFRESH_MS,
            screen=True,
        ) as live:
            while self._running:
                live.update(self._build_layout())
                await asyncio.sleep(refresh_sec)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="forecasts", size=9),
            Layout(name="positions", size=10),
            Layout(name="bottom"),
        )
        layout["bottom"].split_column(
            Layout(name="trades", size=14),
            Layout(name="signals_falcon"),
        )
        layout["signals_falcon"].split_row(
            Layout(name="signals"),
            Layout(name="falcon", size=50),
        )

        layout["header"].update(self._header_panel())
        layout["forecasts"].update(self._forecasts_panel())
        layout["positions"].update(self._positions_panel())
        layout["trades"].update(self._trades_panel())
        layout["signals"].update(self._signals_panel())
        layout["falcon"].update(self._falcon_panel())

        return Panel(layout, title=f"Weather Arb Bot [{self._mode} MODE]", border_style="blue")

    def _header_panel(self) -> Panel:
        s = self._portfolio_stats
        value = s.get("portfolio_value", 0)
        cash = s.get("cash", 0)
        pnl = s.get("daily_pnl", 0)
        dd = s.get("drawdown", 0)
        wr = s.get("win_rate", 0)
        ra = s.get("resolution_accuracy", 0)
        op = s.get("open_positions", 0)
        uptime = s.get("uptime", "0h 0m")
        total = s.get("total_trades", 0)

        pnl_pct = (pnl / max(value - pnl, 1)) * 100 if value > 0 else 0
        pnl_color = "green" if pnl >= 0 else "red"

        text = Text()
        text.append(f"  Portfolio: ${value:.2f}     ", style="bold")
        text.append(f"Daily P&L: ", style="dim")
        text.append(f"${pnl:+.2f} ({pnl_pct:+.1f}%)\n", style=pnl_color)
        text.append(f"  Cash: ${cash:.2f}          ", style="bold")
        text.append(f"Drawdown: {dd*100:.1f}%\n", style="dim")
        text.append(f"  Win Rate: {wr*100:.0f}% ({total})", style="dim")
        text.append(f"    Resolution Accuracy: {ra*100:.0f}%\n", style="dim")
        text.append(f"  Open Positions: {op}     Uptime: {uptime}", style="dim")

        return Panel(text, title="Portfolio", border_style="green")

    def _forecasts_panel(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold cyan")
        table.add_column("City", width=10)
        table.add_column("Date", width=8)
        table.add_column("High", width=6, justify="right")
        table.add_column("sigma", width=5, justify="right")
        table.add_column("Shift", width=7, justify="right")
        table.add_column("Model Run", width=10)
        table.add_column("Agree?", width=6, justify="center")

        for f in self._forecasts[:5]:
            shift = f.get("shift", 0)
            shift_str = f"{shift:+.0f}°F" if shift != 0 else "0°F"
            agree = "Y" if f.get("agreement", True) else "N"
            agree_style = "green" if agree == "Y" else "red"
            table.add_row(
                f.get("city", ""),
                f.get("date", ""),
                f"{f.get('high', 0):.0f}°F",
                f"{f.get('sigma', 0):.1f}",
                shift_str,
                f.get("model_run", ""),
                Text(agree, style=agree_style),
            )

        return Panel(table, title="NOAA Forecasts", border_style="yellow")

    def _positions_panel(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold cyan")
        table.add_column("#", width=3)
        table.add_column("Location", width=10)
        table.add_column("Bucket", width=10)
        table.add_column("Entry", width=7, justify="right")
        table.add_column("Current", width=7, justify="right")
        table.add_column("Size", width=7, justify="right")
        table.add_column("Edge", width=7, justify="right")
        table.add_column("uPnL", width=8, justify="right")

        for i, p in enumerate(self._positions[:8], 1):
            upnl = p.get("unrealized_pnl", 0)
            upnl_style = "green" if upnl >= 0 else "red"
            entry = p.get("entry_price", 0)
            current = p.get("current_price", 0)
            edge = (current - entry) / max(entry, 0.001) * 100
            table.add_row(
                str(i),
                p.get("location", ""),
                p.get("bucket", ""),
                f"${entry:.2f}",
                f"${current:.2f}",
                f"${p.get('size', 0):.2f}",
                f"{edge:+.0f}%",
                Text(f"${upnl:+.2f}", style=upnl_style),
            )

        return Panel(table, title="Open Positions", border_style="magenta")

    def _trades_panel(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold cyan")
        table.add_column("Time", width=8)
        table.add_column("Location", width=10)
        table.add_column("Bucket", width=10)
        table.add_column("Side", width=5)
        table.add_column("Price", width=7, justify="right")
        table.add_column("Size", width=7, justify="right")
        table.add_column("Edge", width=7, justify="right")
        table.add_column("Result", width=8, justify="right")

        for t in self._trades[-10:]:
            side_style = "green" if t.get("side") == "BUY" else "red"
            pnl = t.get("pnl")
            result = f"${pnl:+.2f}" if pnl is not None else "open"
            result_style = "green" if pnl and pnl > 0 else ("red" if pnl and pnl < 0 else "dim")
            table.add_row(
                t.get("time", ""),
                t.get("location", ""),
                t.get("bucket", ""),
                Text(t.get("side", ""), style=side_style),
                f"${t.get('price', 0):.2f}",
                f"${t.get('size', 0):.2f}",
                f"{t.get('edge', 0)*100:+.0f}%",
                Text(result, style=result_style),
            )

        return Panel(table, title="Last 10 Trades", border_style="blue")

    def _signals_panel(self) -> Panel:
        lines = self._signals[-12:] if self._signals else ["No signals yet"]
        text = Text("\n".join(lines))
        return Panel(text, title="Signal Log", border_style="cyan")

    def _falcon_panel(self) -> Panel:
        lines = self._falcon_lines if self._falcon_lines else ["Falcon: waiting for data..."]
        text = Text("\n".join(lines))
        return Panel(text, title="Falcon Intel", border_style="yellow")
