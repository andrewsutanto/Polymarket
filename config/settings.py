"""Centralized configuration for the Polymarket arbitrage scanner.

All tunables live here. Override via environment variables where noted.
"""

import os

# === ENTRY / EXIT THRESHOLDS ===
ENTRY_THRESHOLD: float = 0.15
EXIT_THRESHOLD: float = 0.45
MIN_EDGE: float = 0.03
MIN_CONFIDENCE: float = 0.70
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
GAMMA_POLL_INTERVAL: int = 300
CLOB_POLL_INTERVAL: int = 20
FULL_SCAN_INTERVAL: int = 3600

# === MARKET FILTERS ===
MIN_MARKET_LIQUIDITY: float = 100.0
MIN_MARKET_VOLUME_24H: float = 50.0
MAX_MARKET_SPREAD: float = 0.15
CLOB_TOP_N_TRACKED: int = 50
MAX_TTL_DAYS: int = 90
MIN_TTL_HOURS: int = 2

# === STRATEGY THRESHOLDS ===
IMPLIED_ARB_MIN_EDGE: float = 0.03
IMPLIED_ARB_MAX_VIG: float = 0.04
CROSS_MARKET_MIN_GAP: float = 0.05
VOLUME_SPIKE_ZSCORE: float = 2.5
STALE_HOURS_THRESHOLD: int = 48
LINE_MOVEMENT_LOOKBACK: int = 100
MEAN_REVERSION_ZSCORE_ENTRY: float = 2.0
MEAN_REVERSION_ZSCORE_EXIT: float = 0.5

# === FALCON API ===
FALCON_API_BASE: str = "https://narrative.agent.heisenberg.so/v2"
FALCON_API_TOKEN: str = os.getenv("FALCON_API_TOKEN", "")
FALCON_SMART_MONEY_INTERVAL: int = 60
FALCON_SENTIMENT_INTERVAL: int = 120
FALCON_CROSS_MARKET_INTERVAL: int = 120
FALCON_TOP_TRADERS_COUNT: int = 20
FALCON_ENABLED: bool = os.getenv("FALCON_ENABLED", "false").lower() == "true"

# === POLYMARKET ===
GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
POLYMARKET_HOST: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_SECRET: str = os.getenv("POLYMARKET_SECRET", "")
POLYMARKET_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", "")
POLYMARKET_CHAIN_ID: int = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
POLYMARKET_PRIVATE_KEY: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")

# === EXECUTION ===
ORDER_TIMEOUT: int = 30
SIMULATED_SLIPPAGE: float = 0.005
USER_AGENT: str = "(polymarket-scanner, contact@example.com)"

# === INFRASTRUCTURE ===
DB_PATH: str = "data/trades.db"
SNAPSHOT_INTERVAL: int = 60
LOG_DIR: str = "logs"

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_RATE_LIMIT: int = 20

# === SAFETY ===
DRAWDOWN_WARNING_LEVELS: list[float] = [0.10, 0.15]
LIVE_TRADING_ENABLED: bool = False
