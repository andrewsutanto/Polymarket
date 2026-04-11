# Alpha Research Pipeline Log

Automated daily research + simulation pipeline. Entries added by scheduled agents.

---

## 2026-04-10 — Pipeline Initialized

**Setup:**
- Daily research agent: 9:00 AM — datamine → research → simulate → report
- Sim monitor: every 4 hours — check GH Actions results, log new findings

**Current state:**
- 40 commits on GitHub
- 7 research documents (4,652 lines)
- Calibration strategy replacing broken Markov model
- Wallet copytrade system built
- Early exits (TP/SL/time) implemented
- Parallel strategy comparison running on GH Actions

**Pending research to implement:**
- Residual RSI (price minus calibration fair value)
- Multi-timescale volume Z-score (5x on 6h baseline)
- PMVWAP mean reversion (24h rolling)
- Fixed-cent profit targets (per exit research)
- Model-based exits (edge-gone detection)

## 2026-04-11 — Automated Pipeline Run

**New PMXT files:** 9

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           995.0
  Unrealized Pnl                 -0.03
  Total Value                    999.97
  Return Pct                     -0.0
  Total Trades                   1
  Resolved                       0
  Open Positions                 1
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               50.0
  Max Slippage Bps               50.0
  Sim Duration                   2026-04-10 08:59:58 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           985.19
  Unrealized Pnl                 -0.37
  Total Value                    999.82
  Return Pct                     -0.0
  Total Trades                   7
  Resolved                       4
  Open Positions                 3
  Wins                           2
  Losses                         2
  Win Rate                       50.0
  Total Pnl                      0.19
  Avg Pnl                        0.05
```
### Wallet screening
```
09:37:21 [INFO] === Polymarket Wallet Screener ===
09:37:21 [INFO]   Min trades: 20
09:37:21 [INFO]   Min win rate: 55%
09:37:21 [INFO]   Min PnL: $1000
09:37:21 [INFO]   Fetch limit: 1000 trades
09:37:21 [INFO]   Top wallets: 20
09:37:21 [INFO]   Markets to scan: 30
09:37:21 [INFO] Fetching active markets...
09:37:21 [INFO] Found 60 token IDs across 30 markets
09:37:27 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:37:33 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:37:39 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:37:45 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:37:51 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:37:57 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:37:57 [INFO] Total raw trades fetched: 0
09:37:57 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```
