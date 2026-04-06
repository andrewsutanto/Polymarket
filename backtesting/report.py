"""Report generation: Rich terminal + interactive HTML (Plotly).

Produces publication-grade performance reports with equity curves,
drawdown charts, strategy attribution, regime timelines, and more.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backtesting.metrics import MetricsReport
from backtesting.backtest_engine import BacktestResult
from backtesting.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def print_terminal_report(
    is_report: MetricsReport | None = None,
    oos_report: MetricsReport | None = None,
    result: BacktestResult | None = None,
    wf_result: WalkForwardResult | None = None,
) -> None:
    """Print a Rich terminal report."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    # Header
    console.print(Panel("[bold]BACKTEST REPORT[/bold]", border_style="blue"))

    # IS vs OOS comparison
    if is_report or oos_report:
        table = Table(title="Key Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", width=25)
        if is_report:
            table.add_column("In-Sample", justify="right", width=15)
        if oos_report:
            table.add_column("Out-of-Sample", justify="right", width=15)

        metrics = [
            ("Total Return", "total_return", "{:.2%}"),
            ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
            ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
            ("Calmar Ratio", "calmar_ratio", "{:.2f}"),
            ("Max Drawdown", "max_drawdown", "{:.2%}"),
            ("Win Rate", "win_rate", "{:.2%}"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
            ("# Trades", "n_trades", "{}"),
            ("Avg Win", "avg_win", "${:.4f}"),
            ("Avg Loss", "avg_loss", "${:.4f}"),
            ("Expectancy", "expectancy", "${:.4f}"),
            ("Volatility (ann)", "annualized_volatility", "{:.2%}"),
            ("VaR 95%", "var_95", "{:.4f}"),
        ]

        for label, attr, fmt in metrics:
            row = [label]
            if is_report:
                row.append(fmt.format(getattr(is_report, attr, 0)))
            if oos_report:
                row.append(fmt.format(getattr(oos_report, attr, 0)))
            table.add_row(*row)

        console.print(table)

    # Overfit diagnostics
    if oos_report and oos_report.sharpe_delta != 0:
        console.print()
        ov_table = Table(title="Overfitting Diagnostics", header_style="bold yellow")
        ov_table.add_column("Metric", width=25)
        ov_table.add_column("Value", justify="right", width=15)
        ov_table.add_row("IS Sharpe", f"{oos_report.is_sharpe:.2f}")
        ov_table.add_row("OOS Sharpe", f"{oos_report.oos_sharpe:.2f}")
        ov_table.add_row("Sharpe Delta", f"{oos_report.sharpe_delta:.2f}")
        ov_table.add_row("Deflated Sharpe", f"{oos_report.deflated_sharpe:.3f}")
        flag = "[red]YES[/red]" if oos_report.overfit_flag else "[green]NO[/green]"
        ov_table.add_row("Overfit Flag (>30%)", flag)
        console.print(ov_table)

    # Strategy attribution
    report = oos_report or is_report
    if report and report.strategy_pnl:
        console.print()
        s_table = Table(title="Strategy Attribution", header_style="bold magenta")
        s_table.add_column("Strategy", width=20)
        s_table.add_column("P&L", justify="right", width=10)
        s_table.add_column("Trades", justify="right", width=8)
        s_table.add_column("Win Rate", justify="right", width=10)
        for s in report.strategy_pnl:
            s_table.add_row(
                s,
                f"${report.strategy_pnl[s]:+.4f}",
                str(report.strategy_trades.get(s, 0)),
                f"{report.strategy_win_rates.get(s, 0):.0%}",
            )
        console.print(s_table)

    # Regime breakdown
    if report and report.regime_time_pct:
        console.print()
        r_table = Table(title="Regime Breakdown", header_style="bold green")
        r_table.add_column("Regime", width=20)
        r_table.add_column("Time %", justify="right", width=10)
        r_table.add_column("P&L", justify="right", width=10)
        r_table.add_column("Sharpe", justify="right", width=10)
        for r in report.regime_time_pct:
            r_table.add_row(
                r,
                f"{report.regime_time_pct[r]:.0%}",
                f"${report.regime_pnl.get(r, 0):+.4f}",
                f"{report.regime_sharpe.get(r, 0):.2f}",
            )
        console.print(r_table)

    # City performance
    if report and report.city_pnl:
        console.print()
        c_table = Table(title="City Performance", header_style="bold cyan")
        c_table.add_column("City", width=15)
        c_table.add_column("P&L", justify="right", width=10)
        c_table.add_column("Win Rate", justify="right", width=10)
        for c in report.city_pnl:
            c_table.add_row(c, f"${report.city_pnl[c]:+.4f}", f"{report.city_win_rates.get(c, 0):.0%}")
        console.print(c_table)

    # Walk-forward summary
    if wf_result:
        console.print()
        wf_table = Table(title="Walk-Forward Windows", header_style="bold blue")
        wf_table.add_column("Fold", width=6)
        wf_table.add_column("IS Sharpe", justify="right", width=10)
        wf_table.add_column("OOS Sharpe", justify="right", width=10)
        wf_table.add_column("OOS Return", justify="right", width=10)
        wf_table.add_column("Trades", justify="right", width=8)
        wf_table.add_column("Overfit?", justify="center", width=8)
        for w in wf_result.windows:
            flag = "[red]YES[/red]" if w.overfit_warning else "[green]NO[/green]"
            wf_table.add_row(
                str(w.fold), f"{w.is_sharpe:.2f}", f"{w.oos_sharpe:.2f}",
                f"{w.oos_return:.2%}", str(w.oos_n_trades), flag,
            )
        wf_table.add_row(
            "[bold]AGG[/bold]", "", f"{wf_result.aggregate_sharpe:.2f}",
            f"{wf_result.aggregate_return:.2%}", str(wf_result.aggregate_n_trades), "",
        )
        console.print(wf_table)

    # Top/bottom trades
    if result and result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        console.print()
        t_table = Table(title="Top 5 / Bottom 5 Trades", header_style="bold")
        t_table.add_column("City", width=10)
        t_table.add_column("Strategy", width=16)
        t_table.add_column("Entry", justify="right", width=8)
        t_table.add_column("Exit", justify="right", width=8)
        t_table.add_column("P&L", justify="right", width=10)
        for t in sorted_trades[:5]:
            t_table.add_row(t.city, t.strategy_name, f"${t.entry_price:.3f}", f"${t.exit_price:.3f}",
                            f"[green]${t.pnl:+.4f}[/green]")
        t_table.add_row("---", "---", "---", "---", "---")
        for t in sorted_trades[-5:]:
            t_table.add_row(t.city, t.strategy_name, f"${t.entry_price:.3f}", f"${t.exit_price:.3f}",
                            f"[red]${t.pnl:+.4f}[/red]")
        console.print(t_table)


def generate_html_report(
    is_report: MetricsReport | None = None,
    oos_report: MetricsReport | None = None,
    result: BacktestResult | None = None,
    wf_result: WalkForwardResult | None = None,
    output_path: str | None = None,
) -> str:
    """Generate a self-contained interactive HTML report.

    Returns:
        Path to the generated HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path or str(RESULTS_DIR / "backtest_report.html")

    figs: list[str] = []

    # Equity curve + drawdown
    if result and result.equity_curve:
        ec = np.array(result.equity_curve)
        peak = np.maximum.accumulate(ec)
        dd = (peak - ec) / np.where(peak > 0, peak, 1.0) * -100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            subplot_titles=("Equity Curve", "Drawdown (%)"))
        fig.add_trace(go.Scatter(y=ec.tolist(), mode="lines", name="Equity",
                                 line=dict(color="#e94560")), row=1, col=1)
        fig.add_trace(go.Scatter(y=dd.tolist(), mode="lines", name="Drawdown",
                                 fill="tozeroy", line=dict(color="#ff6b6b")), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", title="Portfolio Performance")
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # Regime timeline
    if result and result.regime_history:
        regime_colors = {
            "ACTIVE_WEATHER": "#e94560", "STABLE": "#4ecca3", "CONSENSUS": "#0f3460",
            "DISAGREEMENT": "#f39c12", "NEUTRAL": "#95a5a6",
        }
        colors = [regime_colors.get(r, "#95a5a6") for r in result.regime_history]
        fig = go.Figure(go.Bar(
            y=["Regime"] * len(result.regime_history),
            x=[1] * len(result.regime_history),
            orientation="h",
            marker_color=colors,
            hovertext=result.regime_history,
        ))
        fig.update_layout(height=100, template="plotly_dark", title="Regime Timeline",
                          showlegend=False, yaxis_visible=False)
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # Trade scatter
    if result and result.trades:
        trades = result.trades
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        fig = go.Figure()
        if wins:
            fig.add_trace(go.Scatter(
                x=[t.holding_bars for t in wins], y=[t.pnl for t in wins],
                mode="markers", name="Win", marker=dict(color="#4ecca3", size=8),
                text=[t.strategy_name for t in wins],
            ))
        if losses:
            fig.add_trace(go.Scatter(
                x=[t.holding_bars for t in losses], y=[t.pnl for t in losses],
                mode="markers", name="Loss", marker=dict(color="#e94560", size=8),
                text=[t.strategy_name for t in losses],
            ))
        fig.update_layout(height=400, template="plotly_dark", title="Trade Returns vs Holding Period",
                          xaxis_title="Holding Bars", yaxis_title="P&L (USD)")
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # Strategy P&L bar chart
    report = oos_report or is_report
    if report and report.strategy_pnl:
        strats = list(report.strategy_pnl.keys())
        pnls = [report.strategy_pnl[s] for s in strats]
        colors = ["#4ecca3" if p >= 0 else "#e94560" for p in pnls]
        fig = go.Figure(go.Bar(x=strats, y=pnls, marker_color=colors))
        fig.update_layout(height=300, template="plotly_dark", title="Strategy P&L Attribution",
                          yaxis_title="P&L (USD)")
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # Walk-forward per-fold bar chart
    if wf_result and wf_result.windows:
        folds = [f"Fold {w.fold}" for w in wf_result.windows]
        is_sr = [w.is_sharpe for w in wf_result.windows]
        oos_sr = [w.oos_sharpe for w in wf_result.windows]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=folds, y=is_sr, name="IS Sharpe", marker_color="#0f3460"))
        fig.add_trace(go.Bar(x=folds, y=oos_sr, name="OOS Sharpe", marker_color="#e94560"))
        fig.update_layout(height=300, template="plotly_dark", title="Walk-Forward: IS vs OOS Sharpe",
                          barmode="group")
        figs.append(fig.to_html(include_plotlyjs=False, full_html=False))

    # Metrics table HTML
    metrics_html = _metrics_table_html(is_report, oos_report)

    # Assemble
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ background: #1a1a2e; color: #eee; font-family: monospace; padding: 20px; }}
h1 {{ color: #e94560; }} h2 {{ color: #4ecca3; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #333; padding: 6px 12px; text-align: right; }}
th {{ background: #16213e; color: #4ecca3; }}
.warn {{ color: #f39c12; }} .good {{ color: #4ecca3; }} .bad {{ color: #e94560; }}
</style></head><body>
<h1>Backtest Report</h1>
{metrics_html}
{''.join(f'<div>{fig}</div>' for fig in figs)}
<p style="color:#666;margin-top:40px">Generated by Weather Arb Backtesting Framework</p>
</body></html>"""

    with open(path, "w") as f:
        f.write(html)

    logger.info("HTML report saved to %s", path)
    return path


def _metrics_table_html(
    is_report: MetricsReport | None,
    oos_report: MetricsReport | None,
) -> str:
    if not is_report and not oos_report:
        return ""

    rows = [
        ("Total Return", "total_return", "{:.2%}"),
        ("Sharpe", "sharpe_ratio", "{:.2f}"),
        ("Sortino", "sortino_ratio", "{:.2f}"),
        ("Calmar", "calmar_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Trades", "n_trades", "{}"),
        ("Expectancy", "expectancy", "${:.4f}"),
    ]

    header = "<tr><th>Metric</th>"
    if is_report:
        header += "<th>In-Sample</th>"
    if oos_report:
        header += "<th>Out-of-Sample</th>"
    header += "</tr>"

    body = ""
    for label, attr, fmt in rows:
        body += f"<tr><td style='text-align:left'>{label}</td>"
        if is_report:
            body += f"<td>{fmt.format(getattr(is_report, attr, 0))}</td>"
        if oos_report:
            body += f"<td>{fmt.format(getattr(oos_report, attr, 0))}</td>"
        body += "</tr>"

    return f"<h2>Key Metrics</h2><table>{header}{body}</table>"
