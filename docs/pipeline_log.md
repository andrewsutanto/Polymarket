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

## 2026-04-12 — Automated Pipeline Run

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
  Sim Duration                   2026-04-11 08:08:54 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           975.0
  Unrealized Pnl                 -2.19
  Total Value                    997.81
  Return Pct                     -0.2
  Total Trades                   5
  Resolved                       0
  Open Positions                 5
  Avg Latency Ms                 311.7
  Max Latency Ms                 496.1
  Avg Slippage Bps               95.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-11 08:08:54 UTC
```
### Wallet screening
```
09:42:32 [INFO] === Polymarket Wallet Screener ===
09:42:32 [INFO]   Min trades: 20
09:42:32 [INFO]   Min win rate: 55%
09:42:32 [INFO]   Min PnL: $1000
09:42:32 [INFO]   Fetch limit: 1000 trades
09:42:32 [INFO]   Top wallets: 20
09:42:32 [INFO]   Markets to scan: 30
09:42:32 [INFO] Fetching active markets...
09:42:32 [INFO] Found 60 token IDs across 30 markets
09:42:39 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:42:45 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:42:51 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:42:58 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:43:04 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:43:11 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:43:11 [INFO] Total raw trades fetched: 0
09:43:11 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-13 — Automated Pipeline Run

**New PMXT files:** 10

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
  Sim Duration                   2026-04-12 08:37:57 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           975.0
  Unrealized Pnl                 0.07
  Total Value                    1000.07
  Return Pct                     0.0
  Total Trades                   5
  Resolved                       0
  Open Positions                 5
  Avg Latency Ms                 246.7
  Max Latency Ms                 489.6
  Avg Slippage Bps               74.0
  Max Slippage Bps               120.0
  Sim Duration                   2026-04-12 08:37:57 UTC
```
### Wallet screening
```
10:29:52 [INFO] === Polymarket Wallet Screener ===
10:29:52 [INFO]   Min trades: 20
10:29:52 [INFO]   Min win rate: 55%
10:29:52 [INFO]   Min PnL: $1000
10:29:52 [INFO]   Fetch limit: 1000 trades
10:29:52 [INFO]   Top wallets: 20
10:29:52 [INFO]   Markets to scan: 30
10:29:52 [INFO] Fetching active markets...
10:29:52 [INFO] Found 60 token IDs across 30 markets
10:29:58 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:30:05 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:30:12 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:30:18 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:30:25 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:30:31 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:30:32 [INFO] Total raw trades fetched: 0
10:30:32 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-14 — Automated Pipeline Run

**New PMXT files:** 10

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           1000.0
  Unrealized Pnl                 0.0
  Total Value                    1000.0
  Return Pct                     0.0
  Total Trades                   0
  Resolved                       0
  Open Positions                 0
  Sim Duration                   2026-04-13 08:36:20 UTC
  Wall Time S                    1160.3
  Events Processed               142505083
  Markets Seen                   26972
======================================================================
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           976.15
  Unrealized Pnl                 -1.59
  Total Value                    994.56
  Return Pct                     -0.5
  Total Trades                   5
  Resolved                       1
  Open Positions                 4
  Wins                           0
  Losses                         1
  Win Rate                       0.0
  Total Pnl                      -3.85
  Avg Pnl                        -3.85
```
### Wallet screening
```
10:13:37 [INFO] === Polymarket Wallet Screener ===
10:13:37 [INFO]   Min trades: 20
10:13:37 [INFO]   Min win rate: 55%
10:13:37 [INFO]   Min PnL: $1000
10:13:37 [INFO]   Fetch limit: 1000 trades
10:13:37 [INFO]   Top wallets: 20
10:13:37 [INFO]   Markets to scan: 30
10:13:37 [INFO] Fetching active markets...
10:13:37 [INFO] Found 60 token IDs across 30 markets
10:13:43 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:13:50 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:13:56 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:14:03 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:14:09 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:14:16 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:14:16 [INFO] Total raw trades fetched: 0
10:14:16 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-15 — Automated Pipeline Run

**New PMXT files:** 10

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
  Sim Duration                   2026-04-14 08:57:52 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           977.7
  Unrealized Pnl                 -1.28
  Total Value                    1001.42
  Return Pct                     0.1
  Total Trades                   6
  Resolved                       1
  Open Positions                 5
  Wins                           1
  Losses                         0
  Win Rate                       100.0
  Total Pnl                      2.7
  Avg Pnl                        2.7
```
### Wallet screening
```
10:14:52 [INFO] === Polymarket Wallet Screener ===
10:14:52 [INFO]   Min trades: 20
10:14:52 [INFO]   Min win rate: 55%
10:14:52 [INFO]   Min PnL: $1000
10:14:52 [INFO]   Fetch limit: 1000 trades
10:14:52 [INFO]   Top wallets: 20
10:14:52 [INFO]   Markets to scan: 30
10:14:52 [INFO] Fetching active markets...
10:14:52 [INFO] Found 60 token IDs across 30 markets
10:14:58 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:15:05 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:15:11 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:15:18 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:15:24 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:15:31 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:15:31 [INFO] Total raw trades fetched: 0
10:15:31 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-16 — Automated Pipeline Run

**New PMXT files:** 6

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.06
  Total Value                    999.94
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               50.0
  Max Slippage Bps               50.0
  Sim Duration                   2026-04-15 08:13:35 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           985.0
  Unrealized Pnl                 -1.19
  Total Value                    998.81
  Return Pct                     -0.1
  Total Trades                   3
  Resolved                       0
  Open Positions                 3
  Avg Latency Ms                 391.1
  Max Latency Ms                 496.0
  Avg Slippage Bps               66.7
  Max Slippage Bps               100.0
  Sim Duration                   2026-04-15 08:13:35 UTC
```
### Wallet screening
```
10:13:42 [INFO] === Polymarket Wallet Screener ===
10:13:42 [INFO]   Min trades: 20
10:13:42 [INFO]   Min win rate: 55%
10:13:42 [INFO]   Min PnL: $1000
10:13:42 [INFO]   Fetch limit: 1000 trades
10:13:42 [INFO]   Top wallets: 20
10:13:42 [INFO]   Markets to scan: 30
10:13:43 [INFO] Fetching active markets...
10:13:43 [INFO] Found 60 token IDs across 30 markets
10:13:48 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:13:54 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:14:00 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:14:06 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:14:12 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:14:18 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:14:19 [INFO] Total raw trades fetched: 0
10:14:19 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```
