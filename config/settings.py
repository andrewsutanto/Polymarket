"""Centralized configuration for the weather arbitrage bot.

All tunables live here. Override via environment variables where noted.
"""

import os

# === ENTRY / EXIT THRESHOLDS ===
ENTRY_THRESHOLD: float = 0.15
EXIT_THRESHOLD: float = 0.45
MIN_VALUE_RATIO: float = 3.0
MIN_EDGE: float = 0.10
MIN_CONFIDENCE: float = 0.85
STOP_LOSS_RATIO: float = 0.5

# === POSITION SIZING ===
KELLY_FRACTION: float = 0.5
MIN_TRADE_SIZE: float = 0.50
MAX_TRADE_SIZE: float = 5.00
MAX_POSITION_USD: float = 5.00
MAX_OPEN_POSITIONS: int = 15
MAX_TRADES_PER_RUN: int = 5
MAX_DAILY_DRAWDOWN: float = 0.20

# === PORTFOLIO ===
STARTING_CAPITAL: float = 50.0

# === SCAN TIMING (seconds) ===
SCAN_INTERVAL: int = 120
NOAA_HOURLY_INTERVAL: int = 120
NOAA_DAILY_INTERVAL: int = 600
NOAA_GRID_INTERVAL: int = 600
OPEN_METEO_INTERVAL: int = 600
POLYMARKET_POLL_INTERVAL: int = 30
CONTRACT_SCAN_INTERVAL: int = 300

# === FORECAST MODEL ===
FORECAST_SIGMA: dict[int, float] = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}
FORECAST_SIGMA_DEFAULT: float = 5.5
SIGMA_DISAGREE_PENALTY: float = 1.0
SIGMA_STABILITY_BONUS: float = 0.5
FORECAST_SHIFT_ALERT_F: float = 2.0
MAX_LEAD_DAYS: int = 5

# === FALCON API ===
FALCON_API_BASE: str = "https://narrative.agent.heisenberg.so/v2"
FALCON_API_TOKEN: str = os.getenv("FALCON_API_TOKEN", "")
FALCON_SMART_MONEY_INTERVAL: int = 60
FALCON_SENTIMENT_INTERVAL: int = 120
FALCON_CROSS_MARKET_INTERVAL: int = 120
FALCON_TOP_TRADERS_COUNT: int = 20
FALCON_ENABLED: bool = os.getenv("FALCON_ENABLED", "true").lower() == "true"

# === NOAA API ===
NWS_API_BASE: str = "https://api.weather.gov"
NWS_USER_AGENT: str = "(weather-arb-bot, contact@example.com)"
OPEN_METEO_BASE: str = "https://api.open-meteo.com/v1/forecast"

# === POLYMARKET ===
POLYMARKET_HOST: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET: str = os.getenv("POLYMARKET_SECRET", "")
POLYMARKET_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", "")
POLYMARKET_CHAIN_ID: int = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")

# === EXECUTION ===
ORDER_TIMEOUT: int = 30
SIMULATED_SLIPPAGE: float = 0.005
MIN_TTL_HOURS: int = 6

# === INFRASTRUCTURE ===
DB_PATH: str = "data/trades.db"
SNAPSHOT_INTERVAL: int = 60
DASHBOARD_REFRESH_MS: int = 1000
LOG_DIR: str = "logs"

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_RATE_LIMIT: int = 20

# === SAFETY ===
DRAWDOWN_WARNING_LEVELS: list[float] = [0.10, 0.15]
LIVE_TRADING_ENABLED: bool = False
