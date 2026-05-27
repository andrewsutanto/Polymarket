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

## 2026-04-17 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:12:25 [INFO] === Polymarket Wallet Screener ===
10:12:25 [INFO]   Min trades: 20
10:12:25 [INFO]   Min win rate: 55%
10:12:25 [INFO]   Min PnL: $1000
10:12:25 [INFO]   Fetch limit: 1000 trades
10:12:25 [INFO]   Top wallets: 20
10:12:25 [INFO]   Markets to scan: 30
10:12:25 [INFO] Fetching active markets...
10:12:25 [INFO] Found 60 token IDs across 30 markets
10:12:31 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:12:37 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:12:44 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:12:50 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:12:56 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:13:03 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:13:03 [INFO] Total raw trades fetched: 0
10:13:03 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-18 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
09:44:02 [INFO] === Polymarket Wallet Screener ===
09:44:02 [INFO]   Min trades: 20
09:44:02 [INFO]   Min win rate: 55%
09:44:02 [INFO]   Min PnL: $1000
09:44:02 [INFO]   Fetch limit: 1000 trades
09:44:02 [INFO]   Top wallets: 20
09:44:02 [INFO]   Markets to scan: 30
09:44:02 [INFO] Fetching active markets...
09:44:02 [INFO] Found 60 token IDs across 30 markets
09:44:08 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:44:15 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:44:21 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:44:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:44:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:44:41 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:44:41 [INFO] Total raw trades fetched: 0
09:44:41 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-19 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
09:43:42 [INFO] === Polymarket Wallet Screener ===
09:43:42 [INFO]   Min trades: 20
09:43:42 [INFO]   Min win rate: 55%
09:43:42 [INFO]   Min PnL: $1000
09:43:42 [INFO]   Fetch limit: 1000 trades
09:43:42 [INFO]   Top wallets: 20
09:43:42 [INFO]   Markets to scan: 30
09:43:42 [INFO] Fetching active markets...
09:43:42 [INFO] Found 60 token IDs across 30 markets
09:43:47 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:43:53 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:43:59 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:44:05 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:44:11 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:44:17 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:44:18 [INFO] Total raw trades fetched: 0
09:44:18 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-20 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:44:58 [INFO] === Polymarket Wallet Screener ===
10:44:58 [INFO]   Min trades: 20
10:44:58 [INFO]   Min win rate: 55%
10:44:58 [INFO]   Min PnL: $1000
10:44:58 [INFO]   Fetch limit: 1000 trades
10:44:58 [INFO]   Top wallets: 20
10:44:58 [INFO]   Markets to scan: 30
10:44:58 [INFO] Fetching active markets...
10:44:58 [INFO] Found 60 token IDs across 30 markets
10:45:04 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:45:10 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:45:16 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:45:22 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:45:28 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:45:34 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:45:34 [INFO] Total raw trades fetched: 0
10:45:34 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-21 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:17:16 [INFO] === Polymarket Wallet Screener ===
10:17:16 [INFO]   Min trades: 20
10:17:16 [INFO]   Min win rate: 55%
10:17:16 [INFO]   Min PnL: $1000
10:17:16 [INFO]   Fetch limit: 1000 trades
10:17:16 [INFO]   Top wallets: 20
10:17:16 [INFO]   Markets to scan: 30
10:17:16 [INFO] Fetching active markets...
10:17:17 [INFO] Found 60 token IDs across 30 markets
10:17:22 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:17:28 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:17:34 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:17:40 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:17:46 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:17:52 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:17:52 [INFO] Total raw trades fetched: 0
10:17:52 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-22 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:17:37 [INFO] === Polymarket Wallet Screener ===
10:17:37 [INFO]   Min trades: 20
10:17:37 [INFO]   Min win rate: 55%
10:17:37 [INFO]   Min PnL: $1000
10:17:37 [INFO]   Fetch limit: 1000 trades
10:17:37 [INFO]   Top wallets: 20
10:17:37 [INFO]   Markets to scan: 30
10:17:37 [INFO] Fetching active markets...
10:17:37 [INFO] Found 60 token IDs across 30 markets
10:17:43 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:17:50 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:17:56 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:18:03 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:18:09 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:18:16 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:18:16 [INFO] Total raw trades fetched: 0
10:18:16 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-23 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:19:41 [INFO] === Polymarket Wallet Screener ===
10:19:41 [INFO]   Min trades: 20
10:19:41 [INFO]   Min win rate: 55%
10:19:41 [INFO]   Min PnL: $1000
10:19:41 [INFO]   Fetch limit: 1000 trades
10:19:41 [INFO]   Top wallets: 20
10:19:41 [INFO]   Markets to scan: 30
10:19:41 [INFO] Fetching active markets...
10:19:41 [INFO] Found 60 token IDs across 30 markets
10:19:47 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:19:53 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:20:00 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:20:06 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:20:13 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:20:19 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:20:20 [INFO] Total raw trades fetched: 0
10:20:20 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-24 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:18:41 [INFO] === Polymarket Wallet Screener ===
10:18:41 [INFO]   Min trades: 20
10:18:41 [INFO]   Min win rate: 55%
10:18:41 [INFO]   Min PnL: $1000
10:18:41 [INFO]   Fetch limit: 1000 trades
10:18:41 [INFO]   Top wallets: 20
10:18:41 [INFO]   Markets to scan: 30
10:18:41 [INFO] Fetching active markets...
10:18:41 [INFO] Found 60 token IDs across 30 markets
10:18:46 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:18:52 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:18:58 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:19:04 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:19:14 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:19:20 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:19:20 [INFO] Total raw trades fetched: 0
10:19:20 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-25 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
09:49:12 [INFO] === Polymarket Wallet Screener ===
09:49:12 [INFO]   Min trades: 20
09:49:12 [INFO]   Min win rate: 55%
09:49:12 [INFO]   Min PnL: $1000
09:49:12 [INFO]   Fetch limit: 1000 trades
09:49:12 [INFO]   Top wallets: 20
09:49:12 [INFO]   Markets to scan: 30
09:49:12 [INFO] Fetching active markets...
09:49:12 [INFO] Found 60 token IDs across 30 markets
09:49:18 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:49:24 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:49:30 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:49:36 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:49:42 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:49:48 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:49:48 [INFO] Total raw trades fetched: 0
09:49:48 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-26 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
09:52:00 [INFO] === Polymarket Wallet Screener ===
09:52:00 [INFO]   Min trades: 20
09:52:00 [INFO]   Min win rate: 55%
09:52:00 [INFO]   Min PnL: $1000
09:52:00 [INFO]   Fetch limit: 1000 trades
09:52:00 [INFO]   Top wallets: 20
09:52:00 [INFO]   Markets to scan: 30
09:52:00 [INFO] Fetching active markets...
09:52:00 [INFO] Found 60 token IDs across 30 markets
09:52:06 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:52:12 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:52:18 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:52:24 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:52:30 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:52:36 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:52:37 [INFO] Total raw trades fetched: 0
09:52:37 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-27 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:00:24 [INFO] === Polymarket Wallet Screener ===
11:00:24 [INFO]   Min trades: 20
11:00:24 [INFO]   Min win rate: 55%
11:00:24 [INFO]   Min PnL: $1000
11:00:24 [INFO]   Fetch limit: 1000 trades
11:00:24 [INFO]   Top wallets: 20
11:00:24 [INFO]   Markets to scan: 30
11:00:24 [INFO] Fetching active markets...
11:00:24 [INFO] Found 60 token IDs across 30 markets
11:00:29 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:00:35 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:00:41 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:00:47 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:00:53 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:01:00 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:01:00 [INFO] Total raw trades fetched: 0
11:01:00 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-28 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:02:04 [INFO] === Polymarket Wallet Screener ===
11:02:04 [INFO]   Min trades: 20
11:02:04 [INFO]   Min win rate: 55%
11:02:04 [INFO]   Min PnL: $1000
11:02:04 [INFO]   Fetch limit: 1000 trades
11:02:04 [INFO]   Top wallets: 20
11:02:04 [INFO]   Markets to scan: 30
11:02:04 [INFO] Fetching active markets...
11:02:04 [INFO] Found 60 token IDs across 30 markets
11:02:10 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:02:16 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:02:22 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:02:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:02:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:02:40 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:02:40 [INFO] Total raw trades fetched: 0
11:02:40 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-29 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:52:49 [INFO] === Polymarket Wallet Screener ===
10:52:49 [INFO]   Min trades: 20
10:52:49 [INFO]   Min win rate: 55%
10:52:49 [INFO]   Min PnL: $1000
10:52:49 [INFO]   Fetch limit: 1000 trades
10:52:49 [INFO]   Top wallets: 20
10:52:49 [INFO]   Markets to scan: 30
10:52:49 [INFO] Fetching active markets...
10:52:49 [INFO] Found 60 token IDs across 30 markets
10:52:55 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:53:01 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:53:07 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:53:13 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:53:19 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:53:25 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:53:26 [INFO] Total raw trades fetched: 0
10:53:26 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-04-30 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:52:02 [INFO] === Polymarket Wallet Screener ===
10:52:02 [INFO]   Min trades: 20
10:52:02 [INFO]   Min win rate: 55%
10:52:02 [INFO]   Min PnL: $1000
10:52:02 [INFO]   Fetch limit: 1000 trades
10:52:02 [INFO]   Top wallets: 20
10:52:02 [INFO]   Markets to scan: 30
10:52:02 [INFO] Fetching active markets...
10:52:03 [INFO] Found 60 token IDs across 30 markets
10:52:09 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:52:15 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:52:21 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:52:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:52:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:52:41 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:52:41 [INFO] Total raw trades fetched: 0
10:52:41 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-01 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:21:03 [INFO] === Polymarket Wallet Screener ===
10:21:03 [INFO]   Min trades: 20
10:21:03 [INFO]   Min win rate: 55%
10:21:03 [INFO]   Min PnL: $1000
10:21:03 [INFO]   Fetch limit: 1000 trades
10:21:03 [INFO]   Top wallets: 20
10:21:03 [INFO]   Markets to scan: 30
10:21:03 [INFO] Fetching active markets...
10:21:03 [INFO] Found 60 token IDs across 30 markets
10:21:08 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:21:14 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:21:21 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:21:27 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:21:32 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:21:38 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:21:39 [INFO] Total raw trades fetched: 0
10:21:39 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-02 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
09:57:01 [INFO] === Polymarket Wallet Screener ===
09:57:01 [INFO]   Min trades: 20
09:57:01 [INFO]   Min win rate: 55%
09:57:01 [INFO]   Min PnL: $1000
09:57:01 [INFO]   Fetch limit: 1000 trades
09:57:01 [INFO]   Top wallets: 20
09:57:01 [INFO]   Markets to scan: 30
09:57:01 [INFO] Fetching active markets...
09:57:01 [INFO] Found 60 token IDs across 30 markets
09:57:07 [INFO]   Fetched 10/60 tokens, 0 trades so far...
09:57:13 [INFO]   Fetched 20/60 tokens, 0 trades so far...
09:57:19 [INFO]   Fetched 30/60 tokens, 0 trades so far...
09:57:25 [INFO]   Fetched 40/60 tokens, 0 trades so far...
09:57:31 [INFO]   Fetched 50/60 tokens, 0 trades so far...
09:57:37 [INFO]   Fetched 60/60 tokens, 0 trades so far...
09:57:38 [INFO] Total raw trades fetched: 0
09:57:38 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-03 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:01:34 [INFO] === Polymarket Wallet Screener ===
10:01:34 [INFO]   Min trades: 20
10:01:34 [INFO]   Min win rate: 55%
10:01:34 [INFO]   Min PnL: $1000
10:01:34 [INFO]   Fetch limit: 1000 trades
10:01:34 [INFO]   Top wallets: 20
10:01:34 [INFO]   Markets to scan: 30
10:01:34 [INFO] Fetching active markets...
10:01:34 [INFO] Found 60 token IDs across 30 markets
10:01:39 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:01:45 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:01:51 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:01:57 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:02:04 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:02:10 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:02:10 [INFO] Total raw trades fetched: 0
10:02:10 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-04 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:03:24 [INFO] === Polymarket Wallet Screener ===
11:03:24 [INFO]   Min trades: 20
11:03:24 [INFO]   Min win rate: 55%
11:03:24 [INFO]   Min PnL: $1000
11:03:24 [INFO]   Fetch limit: 1000 trades
11:03:24 [INFO]   Top wallets: 20
11:03:24 [INFO]   Markets to scan: 30
11:03:24 [INFO] Fetching active markets...
11:03:24 [INFO] Found 60 token IDs across 30 markets
11:03:29 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:03:35 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:03:41 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:03:47 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:03:53 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:03:59 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:04:00 [INFO] Total raw trades fetched: 0
11:04:00 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-05 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:43:19 [INFO] === Polymarket Wallet Screener ===
10:43:19 [INFO]   Min trades: 20
10:43:19 [INFO]   Min win rate: 55%
10:43:19 [INFO]   Min PnL: $1000
10:43:19 [INFO]   Fetch limit: 1000 trades
10:43:19 [INFO]   Top wallets: 20
10:43:19 [INFO]   Markets to scan: 30
10:43:19 [INFO] Fetching active markets...
10:43:19 [INFO] Found 60 token IDs across 30 markets
10:43:25 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:43:32 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:43:38 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:43:44 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:43:51 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:43:57 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:43:58 [INFO] Total raw trades fetched: 0
10:43:58 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-06 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:03:45 [INFO] === Polymarket Wallet Screener ===
11:03:45 [INFO]   Min trades: 20
11:03:45 [INFO]   Min win rate: 55%
11:03:45 [INFO]   Min PnL: $1000
11:03:45 [INFO]   Fetch limit: 1000 trades
11:03:45 [INFO]   Top wallets: 20
11:03:45 [INFO]   Markets to scan: 30
11:03:45 [INFO] Fetching active markets...
11:03:45 [INFO] Found 60 token IDs across 30 markets
11:03:51 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:03:57 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:04:03 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:04:09 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:04:15 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:04:21 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:04:21 [INFO] Total raw trades fetched: 0
11:04:21 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-07 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:08:02 [INFO] === Polymarket Wallet Screener ===
11:08:02 [INFO]   Min trades: 20
11:08:02 [INFO]   Min win rate: 55%
11:08:02 [INFO]   Min PnL: $1000
11:08:02 [INFO]   Fetch limit: 1000 trades
11:08:02 [INFO]   Top wallets: 20
11:08:02 [INFO]   Markets to scan: 30
11:08:02 [INFO] Fetching active markets...
11:08:02 [INFO] Found 60 token IDs across 30 markets
11:08:08 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:08:15 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:08:21 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:08:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:08:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:08:41 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:08:41 [INFO] Total raw trades fetched: 0
11:08:41 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-08 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:22:29 [INFO] === Polymarket Wallet Screener ===
10:22:29 [INFO]   Min trades: 20
10:22:29 [INFO]   Min win rate: 55%
10:22:29 [INFO]   Min PnL: $1000
10:22:29 [INFO]   Fetch limit: 1000 trades
10:22:29 [INFO]   Top wallets: 20
10:22:29 [INFO]   Markets to scan: 30
10:22:29 [INFO] Fetching active markets...
10:22:29 [INFO] Found 60 token IDs across 30 markets
10:22:35 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:22:42 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:22:48 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:22:55 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:23:01 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:23:07 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:23:08 [INFO] Total raw trades fetched: 0
10:23:08 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-09 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:03:44 [INFO] === Polymarket Wallet Screener ===
10:03:44 [INFO]   Min trades: 20
10:03:44 [INFO]   Min win rate: 55%
10:03:44 [INFO]   Min PnL: $1000
10:03:44 [INFO]   Fetch limit: 1000 trades
10:03:44 [INFO]   Top wallets: 20
10:03:44 [INFO]   Markets to scan: 30
10:03:44 [INFO] Fetching active markets...
10:03:44 [INFO] Found 60 token IDs across 30 markets
10:03:50 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:03:56 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:04:02 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:04:08 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:04:14 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:04:20 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:04:20 [INFO] Total raw trades fetched: 0
10:04:20 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-10 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:11:04 [INFO] === Polymarket Wallet Screener ===
10:11:04 [INFO]   Min trades: 20
10:11:04 [INFO]   Min win rate: 55%
10:11:04 [INFO]   Min PnL: $1000
10:11:04 [INFO]   Fetch limit: 1000 trades
10:11:04 [INFO]   Top wallets: 20
10:11:04 [INFO]   Markets to scan: 30
10:11:04 [INFO] Fetching active markets...
10:11:04 [INFO] Found 60 token IDs across 30 markets
10:11:10 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:11:16 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:11:22 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:11:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:11:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:11:40 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:11:40 [INFO] Total raw trades fetched: 0
10:11:40 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-11 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
12:08:27 [INFO] === Polymarket Wallet Screener ===
12:08:27 [INFO]   Min trades: 20
12:08:27 [INFO]   Min win rate: 55%
12:08:27 [INFO]   Min PnL: $1000
12:08:27 [INFO]   Fetch limit: 1000 trades
12:08:27 [INFO]   Top wallets: 20
12:08:27 [INFO]   Markets to scan: 30
12:08:27 [INFO] Fetching active markets...
12:08:27 [INFO] Found 60 token IDs across 30 markets
12:08:32 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:08:39 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:08:45 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:08:51 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:08:57 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:09:03 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:09:04 [INFO] Total raw trades fetched: 0
12:09:04 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-12 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:19:05 [INFO] === Polymarket Wallet Screener ===
11:19:05 [INFO]   Min trades: 20
11:19:05 [INFO]   Min win rate: 55%
11:19:05 [INFO]   Min PnL: $1000
11:19:05 [INFO]   Fetch limit: 1000 trades
11:19:05 [INFO]   Top wallets: 20
11:19:05 [INFO]   Markets to scan: 30
11:19:05 [INFO] Fetching active markets...
11:19:05 [INFO] Found 60 token IDs across 30 markets
11:19:10 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:19:16 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:19:22 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:19:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:19:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:19:40 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:19:40 [INFO] Total raw trades fetched: 0
11:19:40 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-13 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:28:58 [INFO] === Polymarket Wallet Screener ===
11:28:58 [INFO]   Min trades: 20
11:28:58 [INFO]   Min win rate: 55%
11:28:58 [INFO]   Min PnL: $1000
11:28:58 [INFO]   Fetch limit: 1000 trades
11:28:58 [INFO]   Top wallets: 20
11:28:58 [INFO]   Markets to scan: 30
11:28:58 [INFO] Fetching active markets...
11:28:58 [INFO] Found 60 token IDs across 30 markets
11:29:03 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:29:09 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:29:15 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:29:21 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:29:27 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:29:33 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:29:33 [INFO] Total raw trades fetched: 0
11:29:33 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-14 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:12:28 [INFO] === Polymarket Wallet Screener ===
11:12:28 [INFO]   Min trades: 20
11:12:28 [INFO]   Min win rate: 55%
11:12:28 [INFO]   Min PnL: $1000
11:12:28 [INFO]   Fetch limit: 1000 trades
11:12:28 [INFO]   Top wallets: 20
11:12:28 [INFO]   Markets to scan: 30
11:12:28 [INFO] Fetching active markets...
11:12:28 [INFO] Found 60 token IDs across 30 markets
11:12:34 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:12:41 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:12:48 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:12:54 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:13:01 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:13:07 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:13:08 [INFO] Total raw trades fetched: 0
11:13:08 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-15 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:19:22 [INFO] === Polymarket Wallet Screener ===
11:19:22 [INFO]   Min trades: 20
11:19:22 [INFO]   Min win rate: 55%
11:19:22 [INFO]   Min PnL: $1000
11:19:22 [INFO]   Fetch limit: 1000 trades
11:19:22 [INFO]   Top wallets: 20
11:19:22 [INFO]   Markets to scan: 30
11:19:22 [INFO] Fetching active markets...
11:19:22 [INFO] Found 60 token IDs across 30 markets
11:19:28 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:19:35 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:19:41 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:19:48 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:19:54 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:20:01 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:20:01 [INFO] Total raw trades fetched: 0
11:20:01 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-16 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:12:05 [INFO] === Polymarket Wallet Screener ===
10:12:05 [INFO]   Min trades: 20
10:12:05 [INFO]   Min win rate: 55%
10:12:05 [INFO]   Min PnL: $1000
10:12:05 [INFO]   Fetch limit: 1000 trades
10:12:05 [INFO]   Top wallets: 20
10:12:05 [INFO]   Markets to scan: 30
10:12:05 [INFO] Fetching active markets...
10:12:05 [INFO] Found 60 token IDs across 30 markets
10:12:10 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:12:16 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:12:22 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:12:28 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:12:34 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:12:40 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:12:41 [INFO] Total raw trades fetched: 0
10:12:41 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-17 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:20:59 [INFO] === Polymarket Wallet Screener ===
10:20:59 [INFO]   Min trades: 20
10:20:59 [INFO]   Min win rate: 55%
10:20:59 [INFO]   Min PnL: $1000
10:20:59 [INFO]   Fetch limit: 1000 trades
10:20:59 [INFO]   Top wallets: 20
10:20:59 [INFO]   Markets to scan: 30
10:20:59 [INFO] Fetching active markets...
10:20:59 [INFO] Found 60 token IDs across 30 markets
10:21:04 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:21:10 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:21:16 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:21:22 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:21:28 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:21:34 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:21:35 [INFO] Total raw trades fetched: 0
10:21:35 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-18 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
12:26:32 [INFO] === Polymarket Wallet Screener ===
12:26:32 [INFO]   Min trades: 20
12:26:32 [INFO]   Min win rate: 55%
12:26:32 [INFO]   Min PnL: $1000
12:26:32 [INFO]   Fetch limit: 1000 trades
12:26:32 [INFO]   Top wallets: 20
12:26:32 [INFO]   Markets to scan: 30
12:26:32 [INFO] Fetching active markets...
12:26:33 [INFO] Found 60 token IDs across 30 markets
12:26:39 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:26:45 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:26:51 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:26:58 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:27:04 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:27:10 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:27:11 [INFO] Total raw trades fetched: 0
12:27:11 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-19 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:59:49 [INFO] === Polymarket Wallet Screener ===
11:59:49 [INFO]   Min trades: 20
11:59:49 [INFO]   Min win rate: 55%
11:59:49 [INFO]   Min PnL: $1000
11:59:49 [INFO]   Fetch limit: 1000 trades
11:59:49 [INFO]   Top wallets: 20
11:59:49 [INFO]   Markets to scan: 30
11:59:49 [INFO] Fetching active markets...
11:59:49 [INFO] Found 60 token IDs across 30 markets
11:59:55 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:00:01 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:00:07 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:00:13 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:00:19 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:00:25 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:00:26 [INFO] Total raw trades fetched: 0
12:00:26 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-20 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:47:45 [INFO] === Polymarket Wallet Screener ===
11:47:45 [INFO]   Min trades: 20
11:47:45 [INFO]   Min win rate: 55%
11:47:45 [INFO]   Min PnL: $1000
11:47:45 [INFO]   Fetch limit: 1000 trades
11:47:45 [INFO]   Top wallets: 20
11:47:45 [INFO]   Markets to scan: 30
11:47:45 [INFO] Fetching active markets...
11:47:46 [INFO] Found 60 token IDs across 30 markets
11:47:52 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:47:58 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:48:04 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:48:11 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:48:17 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:48:24 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:48:24 [INFO] Total raw trades fetched: 0
11:48:24 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-21 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
12:12:18 [INFO] === Polymarket Wallet Screener ===
12:12:18 [INFO]   Min trades: 20
12:12:18 [INFO]   Min win rate: 55%
12:12:18 [INFO]   Min PnL: $1000
12:12:18 [INFO]   Fetch limit: 1000 trades
12:12:18 [INFO]   Top wallets: 20
12:12:18 [INFO]   Markets to scan: 30
12:12:18 [INFO] Fetching active markets...
12:12:18 [INFO] Found 60 token IDs across 30 markets
12:12:24 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:12:31 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:12:37 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:12:44 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:12:50 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:12:57 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:12:57 [INFO] Total raw trades fetched: 0
12:12:57 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-22 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
11:43:32 [INFO] === Polymarket Wallet Screener ===
11:43:32 [INFO]   Min trades: 20
11:43:32 [INFO]   Min win rate: 55%
11:43:32 [INFO]   Min PnL: $1000
11:43:32 [INFO]   Fetch limit: 1000 trades
11:43:32 [INFO]   Top wallets: 20
11:43:32 [INFO]   Markets to scan: 30
11:43:32 [INFO] Fetching active markets...
11:43:33 [INFO] Found 60 token IDs across 30 markets
11:43:39 [INFO]   Fetched 10/60 tokens, 0 trades so far...
11:43:45 [INFO]   Fetched 20/60 tokens, 0 trades so far...
11:43:52 [INFO]   Fetched 30/60 tokens, 0 trades so far...
11:43:58 [INFO]   Fetched 40/60 tokens, 0 trades so far...
11:44:04 [INFO]   Fetched 50/60 tokens, 0 trades so far...
11:44:11 [INFO]   Fetched 60/60 tokens, 0 trades so far...
11:44:11 [INFO] Total raw trades fetched: 0
11:44:11 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-23 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.1
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:24:33 [INFO] === Polymarket Wallet Screener ===
10:24:33 [INFO]   Min trades: 20
10:24:33 [INFO]   Min win rate: 55%
10:24:33 [INFO]   Min PnL: $1000
10:24:33 [INFO]   Fetch limit: 1000 trades
10:24:33 [INFO]   Top wallets: 20
10:24:33 [INFO]   Markets to scan: 30
10:24:33 [INFO] Fetching active markets...
10:24:33 [INFO] Found 60 token IDs across 30 markets
10:24:39 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:24:44 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:24:50 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:24:56 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:25:02 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:25:08 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:25:09 [INFO] Total raw trades fetched: 0
10:25:09 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-24 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
10:28:48 [INFO] === Polymarket Wallet Screener ===
10:28:48 [INFO]   Min trades: 20
10:28:48 [INFO]   Min win rate: 55%
10:28:48 [INFO]   Min PnL: $1000
10:28:48 [INFO]   Fetch limit: 1000 trades
10:28:48 [INFO]   Top wallets: 20
10:28:48 [INFO]   Markets to scan: 30
10:28:48 [INFO] Fetching active markets...
10:28:48 [INFO] Found 60 token IDs across 30 markets
10:28:54 [INFO]   Fetched 10/60 tokens, 0 trades so far...
10:29:00 [INFO]   Fetched 20/60 tokens, 0 trades so far...
10:29:07 [INFO]   Fetched 30/60 tokens, 0 trades so far...
10:29:13 [INFO]   Fetched 40/60 tokens, 0 trades so far...
10:29:19 [INFO]   Fetched 50/60 tokens, 0 trades so far...
10:29:26 [INFO]   Fetched 60/60 tokens, 0 trades so far...
10:29:26 [INFO] Total raw trades fetched: 0
10:29:26 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-25 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
12:25:51 [INFO] === Polymarket Wallet Screener ===
12:25:51 [INFO]   Min trades: 20
12:25:51 [INFO]   Min win rate: 55%
12:25:51 [INFO]   Min PnL: $1000
12:25:51 [INFO]   Fetch limit: 1000 trades
12:25:51 [INFO]   Top wallets: 20
12:25:51 [INFO]   Markets to scan: 30
12:25:51 [INFO] Fetching active markets...
12:25:51 [INFO] Found 60 token IDs across 30 markets
12:25:57 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:26:03 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:26:09 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:26:15 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:26:21 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:26:27 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:26:27 [INFO] Total raw trades fetched: 0
12:26:27 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```

## 2026-05-27 — Automated Pipeline Run

**New PMXT files:** 0

### calibration strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           990.0
  Unrealized Pnl                 -0.09
  Total Value                    999.91
  Return Pct                     -0.0
  Total Trades                   2
  Resolved                       0
  Open Positions                 2
  Avg Latency Ms                 0.0
  Max Latency Ms                 0.1
  Avg Slippage Bps               100.0
  Max Slippage Bps               150.0
  Sim Duration                   2026-04-08 08:50:29 UTC
```
### markov strategy
```
  SIMULATION RESULTS
======================================================================
  Capital                        1000.0
  Cash                           979.86
  Unrealized Pnl                 -1.2
  Total Value                    998.66
  Return Pct                     -0.1
  Total Trades                   6
  Resolved                       2
  Open Positions                 4
  Wins                           1
  Losses                         1
  Win Rate                       50.0
  Total Pnl                      -0.14
  Avg Pnl                        -0.07
```
### Wallet screening
```
12:17:38 [INFO] === Polymarket Wallet Screener ===
12:17:38 [INFO]   Min trades: 20
12:17:38 [INFO]   Min win rate: 55%
12:17:38 [INFO]   Min PnL: $1000
12:17:38 [INFO]   Fetch limit: 1000 trades
12:17:38 [INFO]   Top wallets: 20
12:17:38 [INFO]   Markets to scan: 30
12:17:38 [INFO] Fetching active markets...
12:17:38 [INFO] Found 60 token IDs across 30 markets
12:17:44 [INFO]   Fetched 10/60 tokens, 0 trades so far...
12:17:50 [INFO]   Fetched 20/60 tokens, 0 trades so far...
12:17:56 [INFO]   Fetched 30/60 tokens, 0 trades so far...
12:18:02 [INFO]   Fetched 40/60 tokens, 0 trades so far...
12:18:08 [INFO]   Fetched 50/60 tokens, 0 trades so far...
12:18:14 [INFO]   Fetched 60/60 tokens, 0 trades so far...
12:18:14 [INFO] Total raw trades fetched: 0
12:18:14 [ERROR] No trades returned. The CLOB API may not expose wallet addresses in this endpoint. Consider using Polygon RPC or Dune Analytics as alternative data sources.
```
