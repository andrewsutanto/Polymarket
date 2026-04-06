"""Chart generation for Telegram image messages.

Creates P&L equity curves, forecast probability distributions, and
position summary charts as PNG bytes for sending via Telegram.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

logger = logging.getLogger(__name__)

# Consistent dark theme
PLT_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.3,
}


def _apply_style() -> None:
    plt.rcParams.update(PLT_STYLE)
    plt.rcParams["font.size"] = 10


def generate_pnl_chart(
    snapshots: list[dict[str, Any]],
    title: str = "Portfolio Value",
) -> bytes:
    """Generate a P&L equity curve as PNG bytes.

    Args:
        snapshots: List of dicts with 'timestamp' and 'portfolio_value' keys.
        title: Chart title.

    Returns:
        PNG image bytes.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    if not snapshots:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=14, color="#aaa")
        ax.set_title(title)
    else:
        times = []
        values = []
        for s in snapshots:
            ts = s.get("timestamp", "")
            if isinstance(ts, str):
                try:
                    times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    continue
            elif isinstance(ts, datetime):
                times.append(ts)
            else:
                continue
            values.append(float(s.get("portfolio_value", 0)))

        if times:
            ax.plot(times, values, color="#e94560", linewidth=2)
            ax.fill_between(times, values, alpha=0.15, color="#e94560")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            fig.autofmt_xdate()

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("USD")
        ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_forecast_chart(
    location: str,
    target_date: str,
    forecast_high: float,
    sigma: float,
    buckets: dict[str, tuple[int, int, float]],
) -> bytes:
    """Generate a forecast probability distribution chart.

    Args:
        location: City name.
        target_date: Date string.
        forecast_high: Forecasted high temperature.
        sigma: Standard deviation.
        buckets: {label: (low_f, high_f, probability)}.

    Returns:
        PNG image bytes.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    if not buckets:
        ax.text(0.5, 0.5, "No bucket data", ha="center", va="center", fontsize=14, color="#aaa")
    else:
        labels = []
        probs = []
        for label, (low, high, prob) in sorted(buckets.items(), key=lambda x: x[1][0]):
            labels.append(f"{low}-{high}")
            probs.append(prob * 100)

        colors = []
        for label, (low, high, prob) in sorted(buckets.items(), key=lambda x: x[1][0]):
            if low <= forecast_high <= high:
                colors.append("#e94560")
            else:
                colors.append("#0f3460")

        bars = ax.bar(range(len(labels)), probs, color=colors, edgecolor="#e94560", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Probability (%)")

        # Overlay normal distribution curve
        x_range = np.linspace(
            min(b[0] for b in buckets.values()) - 5,
            max(b[1] for b in buckets.values()) + 5,
            200,
        )
        from scipy.stats import norm
        y_curve = norm.pdf(x_range, loc=forecast_high, scale=max(sigma, 0.5))
        y_curve_scaled = y_curve / max(y_curve) * max(probs) if max(probs) > 0 else y_curve

        ax2 = ax.twinx()
        ax2.plot(
            np.interp(x_range, [b[0] for b in sorted(buckets.values(), key=lambda x: x[0])],
                       range(len(labels))),
            y_curve_scaled,
            color="#e94560", linewidth=1.5, alpha=0.6, linestyle="--",
        )
        ax2.set_yticks([])

    ax.set_title(f"{location} — {target_date} | Forecast: {forecast_high:.0f}°F (σ={sigma:.1f})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_positions_chart(positions: list[dict[str, Any]]) -> bytes:
    """Generate a horizontal bar chart of open positions by unrealized P&L.

    Args:
        positions: List of position dicts with location, bucket, unrealized_pnl.

    Returns:
        PNG image bytes.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, max(3, len(positions) * 0.6 + 1)))

    if not positions:
        ax.text(0.5, 0.5, "No open positions", ha="center", va="center", fontsize=14, color="#aaa")
    else:
        labels = [f"{p['location']} {p['bucket']}" for p in positions]
        pnls = [p.get("unrealized_pnl", 0) for p in positions]
        colors = ["#4ecca3" if p >= 0 else "#e94560" for p in pnls]

        ax.barh(range(len(labels)), pnls, color=colors, edgecolor="#333", height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Unrealized P&L (USD)")
        ax.axvline(x=0, color="#eee", linewidth=0.5)

        for i, (pnl, label) in enumerate(zip(pnls, labels)):
            ax.text(pnl + (0.02 if pnl >= 0 else -0.02), i,
                    f"${pnl:+.2f}", va="center",
                    ha="left" if pnl >= 0 else "right",
                    fontsize=9, color="#eee")

    ax.set_title("Open Positions — Unrealized P&L", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_win_rate_chart(wins: int, losses: int, total_pnl: float) -> bytes:
    """Generate a donut chart showing win/loss rate.

    Args:
        wins: Number of winning trades.
        losses: Number of losing trades.
        total_pnl: Total realized P&L.

    Returns:
        PNG image bytes.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    total = wins + losses
    if total == 0:
        ax.text(0.5, 0.5, "No resolved trades", ha="center", va="center",
                fontsize=14, color="#aaa", transform=ax.transAxes)
    else:
        sizes = [wins, losses]
        colors = ["#4ecca3", "#e94560"]
        labels = [f"Won ({wins})", f"Lost ({losses})"]

        wedges, texts = ax.pie(
            sizes, colors=colors, startangle=90,
            wedgeprops={"width": 0.4, "edgecolor": "#1a1a2e", "linewidth": 2},
        )

        # Center text
        rate = wins / total * 100
        ax.text(0, 0, f"{rate:.0f}%\nWin Rate", ha="center", va="center",
                fontsize=16, fontweight="bold", color="#eee")

        ax.legend(labels, loc="lower center", fontsize=10,
                  frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.05))

    pnl_color = "#4ecca3" if total_pnl >= 0 else "#e94560"
    ax.set_title(f"Performance | P&L: ${total_pnl:+.2f}",
                 fontsize=12, fontweight="bold", color=pnl_color)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
