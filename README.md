# Polymarket Weather Arbitrage Bot

A production-grade Python bot that exploits pricing lag between Polymarket's daily temperature markets and professional weather forecasts from NOAA/NWS.

## How It Works

Weather models (GFS, ECMWF) update on fixed 6-hour cycles, but Polymarket odds don't reprice instantly -- creating a recurring latency window.

1. **NOAA publishes a forecast** -- Bot converts the point forecast + uncertainty into a probability distribution across temperature buckets
2. **Polymarket prices a bucket at $0.08** -- But the NOAA-derived probability says that bucket should be ~$0.40
3. **Bot buys** -- Entry: market price < $0.15 AND model probability > 3x market price
4. **Market reprices** -- Other traders notice, price rises to $0.45+
5. **Bot exits** -- Or holds to resolution ($1.00 payout if correct)

## Architecture

```
config/          - Settings, city configs, API key template
core/
  noaa_feed.py       - NOAA/NWS + Open-Meteo forecast polling
  weather_model.py   - Forecast -> probability distribution (normal CDF)
  polymarket_feed.py - Polymarket CLOB polling + contract discovery
  falcon_intel.py    - Smart money, sentiment, cross-market (Kalshi) data
  signal_engine.py   - Mispricing detection + edge calculation
  risk_manager.py    - Half-Kelly sizing, position limits, kill switch
  executor.py        - Paper + live order execution
  portfolio.py       - Portfolio state, P&L, resolution tracking
infra/
  database.py        - SQLite persistence (trades, forecasts, snapshots)
  telegram.py        - Alert service with command handlers
  dashboard.py       - Rich terminal UI
main.py              - Async orchestrator
```

## Quick Start

```bash
# Clone
git clone https://github.com/andrewsutanto/Polymarket.git
cd Polymarket

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure (optional -- bot runs in paper mode without any keys)
cp config/secrets.py.example config/secrets.py
# Edit config/secrets.py with your API keys

# Run in paper mode (default)
python main.py

# Run without dashboard (headless/logging only)
python main.py --no-dashboard
```

## Configuration

All tunables are in `config/settings.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENTRY_THRESHOLD` | 0.15 | Only buy buckets priced below $0.15 |
| `EXIT_THRESHOLD` | 0.45 | Sell when price rises above $0.45 |
| `MIN_VALUE_RATIO` | 3.0 | Model prob must be 3x+ market price |
| `MIN_CONFIDENCE` | 0.85 | Minimum composite confidence score |
| `KELLY_FRACTION` | 0.5 | Half-Kelly position sizing |
| `MAX_TRADE_SIZE` | 5.00 | Max USD per trade |
| `MAX_OPEN_POSITIONS` | 15 | Max concurrent positions |
| `STARTING_CAPITAL` | 50.00 | Initial paper trading capital |

## Tracked Cities

- New York City (Central Park / KNYC)
- Chicago (O'Hare / KORD)
- Seattle (Sea-Tac / KSEA)
- Atlanta (Hartsfield-Jackson / KATL)
- Dallas (DFW / KDFW)

## Data Sources

- **NOAA/NWS API** -- Free, no key needed. Hourly + daily + raw grid forecasts.
- **Open-Meteo** -- Free second opinion (GFS, ECMWF ensembles).
- **Polymarket CLOB** -- Order book polling for weather temperature contracts.
- **Falcon API** -- Smart money tracking, sentiment, cross-market Kalshi comparison.

## Live Trading

Live mode requires **three independent flags** to prevent accidental deployment:

1. `--live` CLI flag
2. `ENABLE_LIVE_TRADING=true` environment variable
3. `LIVE_TRADING_ENABLED = True` in `config/settings.py`

Plus a 10-second confirmation countdown at startup.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Portfolio state + active forecasts |
| `/kill` | Activate kill switch (halt all trading) |
| `/trades` | Last 10 trades |
| `/pnl` | Daily P&L breakdown |
| `/weather` | Current NOAA forecasts |
| `/positions` | Open positions with current edge |

## Scaling Strategy

1. **Calibration** ($2-5): Paper trade 3-5 days. Analyze resolution accuracy, tune sigma values.
2. **Small live** ($5-20): Go live with tiny positions after calibration shows positive EV.
3. **Scale** ($20-50+): Increase position sizes, add locations, extend to precipitation markets.

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Safety Features

- Paper trading by default
- Triple-flag live trading activation
- Half-Kelly sizing (conservative)
- Kill switch at 20% drawdown (auto + manual via Telegram)
- Drawdown warnings at 10% and 15%
- Max 5 trades per 2-minute scan cycle
- 6-hour minimum time-to-resolution filter
- Graceful shutdown preserves state

## License

MIT
