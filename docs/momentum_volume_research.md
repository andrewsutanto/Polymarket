# Momentum & Volume Indicators for Polymarket Prediction Markets

Research date: 2026-04-10

## Table of Contents

1. [Why Prediction Markets Are Different](#1-why-prediction-markets-are-different)
2. [Momentum Indicators Adapted for Binary Markets](#2-momentum-indicators-adapted-for-binary-markets)
3. [Volume Indicators for Polymarket](#3-volume-indicators-for-polymarket)
4. [Polymarket-Specific Indicators & Trader Practices](#4-polymarket-specific-indicators--trader-practices)
5. [Combined Signals: Momentum + Volume](#5-combined-signals-momentum--volume)
6. [Integration with Existing Codebase](#6-integration-with-existing-codebase)
7. [Top 5 Recommended Indicators](#7-top-5-recommended-indicators)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Why Prediction Markets Are Different

Traditional technical indicators were designed for unbounded equity prices. Polymarket
prices are fundamentally different:

### Key Structural Differences

| Property | Equities | Polymarket |
|---|---|---|
| Price bounds | Unbounded (0 to infinity) | Bounded [0, 1] |
| Resolution | No terminal value | Converges to 0 or 1 at resolution |
| Price driver | Earnings, macro, sentiment | Information arrival about discrete events |
| Volume meaning | Institutional flow, retail | News arrival, informed trading, market making |
| Orderbook depth | Deep (billions) | Thin ($10K-$500K typical) |
| Momentum cause | Trend following, fund flows | Information cascade, slow price discovery |
| Time horizon | Open-ended | Finite (market has end date) |

### Implications for Indicator Design

1. **Bounded normalization required**: RSI and other oscillators that measure 0-100 on
   unbounded prices need re-scaling. A price at 0.95 with RSI 80 means something
   completely different than a stock at RSI 80 -- the PM price SHOULD be near 1.0 if
   the outcome is likely.

2. **Momentum near resolution is informational, not behavioral**: In stocks, momentum
   near earnings is speculative. In PMs, momentum near resolution reflects genuine
   information arrival. Late-stage momentum is MORE reliable, not less.

3. **Volume spikes are news proxies**: In equities, volume spikes can be algorithmic
   rebalancing. In PMs with thin books, a volume spike almost always means new
   information has arrived. Volume Z-score > 5x baseline is a strong news-arrival
   signal.

4. **Kyle's Lambda decays with market maturity**: Academic research on the 2024
   presidential election Polymarket data shows price impact (Kyle's Lambda) declined
   from ~0.518 early to ~0.01 by October -- meaning $1M moved prices 13pp early but
   only 0.25pp late. Indicator sensitivity must adapt to market maturity.

5. **Multiple correlated markets**: Polymarket often has related outcomes (candidate A
   vs candidate B vs candidate C). Momentum in one can predict movement in related
   markets before arbitrage bots catch up (though arb latency is now ~2.7 seconds).

---

## 2. Momentum Indicators Adapted for Binary Markets

### 2.1 RSI (Relative Strength Index) -- Modified for [0,1] Bounds

**Standard RSI problem**: Traditional RSI flags "overbought" at 70 and "oversold" at
30. But a PM price at 0.90 with RSI 80 is not overbought -- it may be correctly
pricing a 90% probability event. The indicator must distinguish between justified
high prices and overreaction.

**Adapted approach -- Residual RSI**:

```
residual_price = market_price - model_fair_value
RSI_residual = RSI(residual_price, period=14)
```

Compute RSI on the RESIDUAL (market price minus calibration-derived fair value),
not on raw price. This way:
- RSI > 70 on residual = market is persistently overshooting fair value (overbought)
- RSI < 30 on residual = market is persistently undershooting (oversold)

**Parameters**:
- Period: 14 bars (using OHLCV at 1h candles from PMXT) or 7 bars for faster signals
- Overbought threshold: 70 (on residual)
- Oversold threshold: 30 (on residual)
- Minimum residual magnitude: 0.03 (below this, RSI noise dominates)

**When it works**: Post-news overreaction. After a headline drops, PM prices
overshoot then revert. Residual RSI > 70 within 30 minutes of a volume spike is
a fade signal with high hit rate in backtests.

**When it fails**: Near resolution when the market should be converging to 0 or 1.
Exclude markets within 24h of resolution.

### 2.2 MACD (Moving Average Convergence Divergence)

**Standard MACD**: MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD.

**Adaptation for PMs**: Standard MACD periods are calibrated for daily stock bars.
PM prices move on news cycles, not trading days. Use shorter periods:

```
MACD_fast = EMA(price, span=6h)
MACD_slow = EMA(price, span=24h)
MACD_line = MACD_fast - MACD_slow
Signal_line = EMA(MACD_line, span=4h)
Histogram = MACD_line - Signal_line
```

**Key signals**:
- **Bullish crossover** (MACD crosses above signal): Price gaining short-term momentum.
  In PMs, this often follows initial news and precedes the market fully pricing it in.
- **Bearish divergence** (price rising but MACD histogram declining): The move is
  losing steam. This is the highest-value MACD signal for PMs -- it catches
  overreactions before they revert.
- **Zero-line cross**: Less useful in PMs because it's slow and markets resolve faster.

**Parameters**:
- Fast EMA: 6 hourly bars (6h)
- Slow EMA: 24 hourly bars (24h)
- Signal EMA: 4 hourly bars (4h)
- Minimum MACD magnitude: 0.005 (filter noise)

**Implementation note**: Use PMXT `fetch_ohlcv()` at 1h resolution. MACD on 1-minute
bars is too noisy for PM prices due to thin books.

### 2.3 Rate of Change (ROC) -- Price Velocity

ROC measures the percentage change over N periods. For PMs, raw ROC is more useful
than percentage ROC because prices are bounded.

```
ROC = price(t) - price(t - N)
```

**Why raw ROC matters for PMs**: A move from 0.50 to 0.55 (+10%) and from 0.90 to
0.95 (+5.6%) are equally significant in absolute terms (0.05 probability shift) but
very different in percentage terms. Use absolute ROC.

**Parameters**:
- Short ROC: N=6 (6 hourly bars) -- captures fast news reaction
- Medium ROC: N=24 (24 hourly bars) -- captures daily trend
- Long ROC: N=72 (72 hourly bars = 3 days) -- captures multi-day drift

**Signals**:
- Short ROC > 0.05 + Medium ROC > 0.03 = Strong momentum, continue holding
- Short ROC < -0.02 + Medium ROC > 0.03 = Momentum divergence, potential reversal
- Short ROC magnitude > 3x historical std = Spike detection (news arrival)

### 2.4 Momentum Divergence (Price vs. Momentum)

The most valuable momentum signal for PMs: price continues moving but the rate of
change is declining.

```
divergence = sign(price_trend) != sign(ROC_trend)
```

Specifically:
1. Compute linear regression slope of price over last 12h
2. Compute linear regression slope of ROC(6h) over last 12h
3. If price slope > 0 but ROC slope < 0 (or vice versa) = bearish/bullish divergence

**This is already partially captured** by the existing `LineMovementStrategy` which
tracks R-squared of price trends. Adding ROC slope comparison strengthens it.

**Parameters**:
- Price trend window: 12 bars (12h)
- ROC period: 6 bars (6h)
- ROC trend window: 12 bars (12h)
- Minimum price trend slope: 0.002/bar
- Minimum divergence magnitude: slope signs must differ AND ROC slope magnitude > 0.001

---

## 3. Volume Indicators for Polymarket

### 3.1 Volume Z-Score -- News Arrival Detection

This is the single most important volume indicator for PMs. Already partially
implemented in `VolumeDivergenceStrategy` but needs enhancement.

**Current implementation**: Z-score threshold of 2.5 on rolling 50-bar window.

**Recommended enhancement**:

```python
# Multi-timescale Z-scores
zscore_1h = (vol_1h - mean_6h) / std_6h
zscore_24h = (vol_24h - mean_7d) / std_7d

# Tiered alerts (from Polyscope scanner approach)
SPIKE_MINOR = 2.0    # Moderate interest increase
SPIKE_MAJOR = 5.0    # Significant news arrival
SPIKE_EXTREME = 10.0 # Breaking news / market-moving event
```

**Real-world calibration**: Polymarket scanners fire spike alerts at 2x, 5x, and
10x baseline volume with a minimum $10,000 volume floor. The 6-hour rolling
baseline with 5x threshold is the most commonly used by profitable traders.

**Documented example**: February 2026, an NFL award market saw bid depth explode
from $9,700 6-hour average to $114,020 in a single collection window -- a clear
smart money accumulation signal detected before price moved.

### 3.2 On-Balance Volume (OBV) -- Adapted

**Standard OBV**: Cumulative sum of volume on up-days minus volume on down-days.

**PM Adaptation**: Use trade-level data, not daily bars. Classify each trade as
buyer-initiated or seller-initiated using trade-and-quote matching:

```python
def compute_obv_pm(trades, mid_prices):
    """OBV adapted for prediction markets using trade classification."""
    obv = 0.0
    obv_series = []
    for trade in trades:
        # Classify: if trade price > mid at time of trade, buyer initiated
        if trade.price > mid_prices[trade.timestamp]:
            obv += trade.size_usd
        else:
            obv -= trade.size_usd
        obv_series.append(obv)
    return obv_series
```

**Signals**:
- OBV rising while price flat = Accumulation (informed buying before price moves)
- OBV falling while price flat = Distribution (informed selling)
- OBV confirming price move = Continuation likely
- OBV diverging from price = Reversal likely

**Parameters**:
- Use hourly OBV aggregation (not per-trade, too noisy)
- OBV trend: linear regression of OBV over last 24h
- Price trend: linear regression of price over last 24h
- Divergence signal: OBV slope and price slope have opposite signs

**Data source**: PMXT `fetch_ohlcv()` includes volume. For trade-level classification,
use CLOB API trade history or PMXT archive parquet files.

### 3.3 VWAP Deviation -- Mean Reversion Signal

**VWAP** (Volume-Weighted Average Price) represents the "fair" execution price.
Deviation from VWAP indicates overextension.

```python
def compute_vwap(prices, volumes):
    """Compute VWAP over a window."""
    return np.sum(prices * volumes) / np.sum(volumes)

def vwap_deviation(current_price, vwap, std_dev):
    """Z-score of current price relative to VWAP."""
    return (current_price - vwap) / max(std_dev, 0.001)
```

**PM-specific considerations**:
- VWAP is most useful for markets trading between 0.20 and 0.80. Near the
  boundaries (< 0.10 or > 0.90), price should deviate from VWAP if the market
  is converging to resolution.
- Use intraday VWAP (reset at UTC midnight) for short-term mean reversion.
- Use rolling 24h VWAP for medium-term fair value.

**Parameters**:
- VWAP window: 24h rolling or daily reset
- Entry: VWAP deviation Z-score > 2.0 (fade the deviation)
- Exit: VWAP deviation Z-score < 0.5 (return to fair value)
- Exclude markets where price is < 0.10 or > 0.90 (resolution convergence zone)
- Minimum volume in VWAP window: $5,000 (below this, VWAP is unreliable)

**Evidence**: In equity markets, VWAP mean reversion produces 0.89pp average edge,
roughly 6x typical transaction costs. PM transaction costs are lower (2% round-trip
taker fee), so the edge-to-cost ratio should be even better.

### 3.4 Accumulation/Distribution -- Smart Money Detection

Already implemented in `core/smart_money.py` as `SmartMoneyDetector`. Current
implementation tracks large orders (>$1,000 USD) with time-decay weighting and
computes flow imbalance.

**Recommended enhancements**:

1. **Multi-threshold classification**:
   ```python
   RETAIL_ORDER = 0       # < $100
   INFORMED_ORDER = 100   # $100 - $1,000
   WHALE_ORDER = 1000     # $1,000 - $10,000
   INSTITUTIONAL = 10000  # > $10,000
   ```

2. **Accumulation pattern detection**: Look for consistent same-direction flow from
   whale-tier orders over multiple collection windows, even if individual orders
   don't spike. Pattern: 5+ whale buys in 6h with < 2 whale sells = accumulation.

3. **VWAP-flow disagreement**: When whale VWAP buy price is significantly below
   current market price, whales see value (already in SmartMoneyDetector but
   could weight this more heavily).

### 3.5 Volume-Price Confirmation

The simplest combined signal: does volume confirm the price move?

```python
def volume_confirms_price(price_change, volume_zscore, threshold=1.5):
    """
    Returns True if high volume confirms a price move (continuation),
    False if price moved on low volume (likely to reverse).
    """
    if abs(price_change) < 0.02:
        return None  # No meaningful price move
    return volume_zscore > threshold
```

**Implementation**:
- Confirmation (high vol + big move): Trend continuation signal, hold position
- Non-confirmation (big move + low vol): Reversal warning, tighten stops
- Spike without move (high vol + flat price): Accumulation/distribution, same as
  volume divergence strategy

---

## 4. Polymarket-Specific Indicators & Trader Practices

### 4.1 What Profitable PM Traders Actually Use

Based on research across trading blogs, on-chain analysis, and PM analytics tools:

**Top edges used by profitable traders (only 7.6% of wallets are profitable)**:

1. **Information speed**: Markets lag breaking news by 30 seconds to several minutes.
   The fastest traders use news APIs, Twitter firehose, and event calendars to trade
   before the market reprices. This is an information edge, not a technical indicator.

2. **Calibration exploitation**: Longshot bias remains the most robust and
   well-documented systematic edge. Already the core of our `CalibrationEdgeStrategy`.
   Research on 292M trades across 327K binary contracts confirms persistent
   miscalibration at price extremes.

3. **Order flow tracking**: On-chain transparency means every Polymarket trade is
   visible on Polygon. Whale tracking (top wallets with >75% win rate) is a
   documented profitable strategy. Our `SmartMoneyDetector` captures this.

4. **Settlement-rule arbitrage**: Trading on resolution rule interpretation rather
   than headline truth. Not a technical indicator but a research edge.

5. **Cross-platform arbitrage**: Price differences between Polymarket, Kalshi, and
   others. Our `CrossMarketArbStrategy` captures this, though profitable arb windows
   have shrunk to ~2.7 seconds average.

### 4.2 PMXT Data Infrastructure

PMXT provides the data layer for implementing all these indicators:

- `fetch_ohlcv(symbol, timeframe, limit)`: Candlestick data at 1m to 1d resolution.
  This is the primary input for all momentum indicators (RSI, MACD, ROC).
- `fetch_order_book(symbol)`: Real-time orderbook for imbalance calculation.
  Already used by `OrderbookImbalanceStrategy`.
- `fetch_trades(symbol, limit)`: Individual trade records for OBV and smart money.
  Input for `SmartMoneyDetector`.
- WebSocket streaming: Real-time orderbook and trade updates for live indicators.
- Archive (archive.pmxt.dev): Hourly parquet snapshots for backtesting all indicators.

### 4.3 Polymarket Analytics Tools in the Ecosystem

- **PolymarketAnalytics.com**: Price charts, volume tracking, trader leaderboards
- **Polyscope (thepolyscope.com)**: Whale trade scanner with volume spike alerts
  (2x, 5x, 10x baseline), large order detection
- **TrendSpider integration**: Polymarket odds as custom indicators on TradingView-
  style charts, with automated alerting
- **Dune Analytics dashboards**: On-chain activity, volume metrics, trader behavior
- **Polymarket trade tracker (GitHub)**: Open-source PnL analysis tool

---

## 5. Combined Signals: Momentum + Volume

### 5.1 Signal Combination Framework

Individual indicators are noisy. The value is in combination. The existing
`AlphaCombiner` already implements the Fundamental Law of Active Management
(IR = IC x sqrt(N)) with per-strategy IC tracking and correlation-based
effective-N computation. New indicators should plug into this framework.

### 5.2 High-Value Signal Combinations

**Combination 1: Volume-Confirmed Momentum (highest expected value)**
```
Entry: ROC(6h) > 0.03 AND volume_zscore(6h) > 2.0 AND orderbook_imbalance > 2.0
Exit:  ROC(6h) reverses sign OR volume_zscore drops below 1.0
```
Rationale: Price moving in a direction with high volume AND orderbook pressure all
confirming = genuine information-driven move that hasn't fully played out.

**Combination 2: Momentum Exhaustion (reversal signal)**
```
Entry: MACD bearish divergence AND volume_zscore declining AND RSI_residual > 70
Exit:  RSI_residual < 50 OR VWAP deviation < 1.0
```
Rationale: Price still high but momentum fading, volume declining, and residual RSI
overextended = the post-news overreaction is peaking, fade it.

**Combination 3: Smart Money Accumulation (pre-move signal)**
```
Entry: OBV_slope > 0 AND price_slope ~= 0 AND whale_flow_imbalance > 0.4
Exit:  Price moves > 0.05 in predicted direction (take profit) OR 24h timeout
```
Rationale: Informed traders are accumulating (OBV rising, whale flow positive) but
price hasn't moved yet. This is the volume divergence signal combined with smart
money flow for higher confidence.

**Combination 4: Calibration + Momentum Confirmation**
```
Entry: calibration_edge > 0.04 AND ROC(24h) in same direction as edge
Exit:  calibration_edge < 0.02 OR ROC reverses
```
Rationale: The calibration edge identifies a mispriced market. Momentum confirming
the direction of the mispricing (price moving toward fair value) means the market
is beginning to correct. This avoids the "too early" problem where calibration edge
exists but the market stays mispriced for days.

### 5.3 What Does NOT Combine Well

- **RSI + MACD without volume**: In PMs, momentum indicators without volume
  confirmation generate too many false signals because thin books cause choppy
  price action that triggers technical signals without real information flow.

- **Multiple momentum indicators without a fundamental anchor**: Stacking RSI +
  MACD + ROC without calibration edge or smart money flow just measures the same
  thing (recent price direction) three ways. This increases signal correlation
  without adding independent information, reducing effective-N.

- **Short-term indicators near resolution**: Within 24h of market resolution,
  momentum indicators break down because price should be converging to 0 or 1.
  Any mean-reversion or overbought signals in this period are wrong by design.

---

## 6. Integration with Existing Codebase

### 6.1 Current Strategy Inventory

| Strategy | Type | Status | Notes |
|---|---|---|---|
| `CalibrationEdgeStrategy` | Calibration | Production | Core alpha, well-tested |
| `VolumeDivergenceStrategy` | Volume | Production | Needs multi-timescale enhancement |
| `OrderbookImbalanceStrategy` | Volume/Flow | Production | Well-designed, decay-weighted |
| `LineMovementStrategy` | Momentum | Production | Needs ROC divergence addition |
| `MeanReversionStrategy` | Mean-reversion | Production | Specific to post-NOAA updates |
| `SmartMoneyDetector` | Flow | Production | Needs multi-tier classification |
| `CrossMarketArbStrategy` | Arbitrage | Production | Edge declining (2.7s windows) |
| `StaleMarketStrategy` | Structural | Production | Orthogonal to momentum/volume |
| `AlphaCombiner` | Combination | Production | Ready for new signals |
| `EnsembleStrategy` | Combination | Production | Static weights, superseded by AlphaCombiner |

### 6.2 What Needs to Be Built vs. Enhanced

**New indicators to build**:
1. `ResidualRSI` -- RSI on (market_price - calibration_fair_value)
2. `AdaptedMACD` -- PM-calibrated MACD with shorter periods
3. `MultiTimeframeROC` -- ROC at 6h/24h/72h with divergence detection
4. `PMVWAP` -- Prediction market VWAP with deviation bands
5. `OBVAdapted` -- Trade-classified OBV for accumulation detection

**Enhancements to existing code**:
1. `VolumeDivergenceStrategy`: Add multi-timescale Z-scores (1h and 24h)
2. `LineMovementStrategy`: Add ROC slope divergence detection
3. `SmartMoneyDetector`: Add multi-tier order classification
4. `MeanReversionStrategy`: Generalize beyond NOAA to use VWAP deviation
5. `AlphaCombiner`: Register and weight new indicators

### 6.3 Data Requirements

All new indicators need OHLCV data at 1h resolution. Sources:
- **Live**: PMXT `fetch_ohlcv()` via WebSocket or polling
- **Backtest**: PMXT archive parquet files at archive.pmxt.dev
- **Existing**: `data/pmxt_parser.py` already handles PMXT data loading

Trade-level data (for OBV and enhanced smart money):
- **Live**: PMXT `fetch_trades()` or CLOB WebSocket
- **Backtest**: PMXT archive trade snapshots
- **Existing**: `core/clob_feed.py` handles CLOB data

---

## 7. Top 5 Recommended Indicators

Ranked by expected value (signal quality x independence from existing strategies):

### Rank 1: Volume Z-Score Enhancement (Multi-Timescale)

**Why**: Already proven in current `VolumeDivergenceStrategy`. Multi-timescale
version catches both fast news spikes (1h) and sustained interest shifts (24h).
Minimal implementation effort, high expected improvement.

**Parameters**:
- Short window: 1h volume vs 6h rolling mean/std
- Long window: 24h volume vs 7d rolling mean/std
- Spike thresholds: 2x (minor), 5x (major), 10x (extreme)
- Minimum volume floor: $10,000

**Entry logic**: Major spike (5x) + price move < 0.02 = accumulation signal.
Extreme spike (10x) + any price move = news arrival, trade momentum.

**Exit logic**: Volume Z-score returns below 1.0 on both timescales.

### Rank 2: Residual RSI (RSI on Calibration Residual)

**Why**: Combines the proven calibration edge with momentum timing. Pure
calibration edge can be early (market stays mispriced for days). Residual RSI
tells you WHEN the market is overextending vs. correcting.

**Parameters**:
- RSI period: 14 hourly bars
- Overbought: 70 (fade signal on residual)
- Oversold: 30 (buy signal on residual)
- Minimum residual: 0.03
- Exclude: Markets within 24h of resolution

**Entry logic**: Residual RSI < 30 AND calibration_edge > 0.04 = strong buy.
Residual RSI > 70 AND calibration_edge < -0.04 = strong sell.

**Exit logic**: Residual RSI crosses 50 (mean reversion target).

### Rank 3: Adapted MACD (Bearish Divergence Focus)

**Why**: MACD bearish divergence (price rising, histogram declining) is the
highest-precision PM signal because it catches the peak of post-news overreaction.
Works best in combination with volume decline.

**Parameters**:
- Fast EMA: 6h
- Slow EMA: 24h
- Signal EMA: 4h
- Divergence detection: Price slope > 0 AND histogram slope < 0 over 6h window
- Minimum histogram magnitude: 0.005

**Entry logic**: MACD bearish divergence detected = fade the current price direction.
Require volume Z-score < 1.5 (volume also declining) for confirmation.

**Exit logic**: MACD histogram reverses direction OR price moves 0.03 in trade direction.

### Rank 4: PMVWAP Mean Reversion

**Why**: VWAP deviation is the one technical indicator with statistically robust
and economically meaningful edge in backtests (0.89pp average in equities, likely
higher in PMs due to thinner books and lower fees). Independent from momentum
signals -- measures overextension from volume-weighted fair value.

**Parameters**:
- VWAP window: 24h rolling
- Entry: Z-score > 2.0 standard deviations from VWAP
- Exit: Z-score < 0.5 (return to VWAP)
- Price filter: Only for markets with price in [0.15, 0.85]
- Volume filter: Minimum $5,000 in VWAP window

**Entry logic**: Price > VWAP + 2*std = sell (overextended above VWAP).
Price < VWAP - 2*std = buy (underextended below VWAP).

**Exit logic**: Price returns within 0.5 std of VWAP.

### Rank 5: OBV-Price Divergence (Accumulation Detection)

**Why**: Detects informed trading before price moves. Independent from all other
signals because it uses trade classification (buyer vs seller initiated) rather
than price or volume magnitude. Combines well with smart money flow.

**Parameters**:
- OBV aggregation: Hourly
- OBV trend: Linear regression slope over 24h
- Price trend: Linear regression slope over 24h
- Divergence signal: OBV slope and price slope have opposite signs
- Minimum OBV slope magnitude: $500/hour
- Minimum divergence duration: 6h (avoid transient divergences)

**Entry logic**: OBV slope > $500/h AND price slope < 0 (or flat) over 6h+ =
accumulation, buy. OBV slope < -$500/h AND price slope > 0 (or flat) = distribution,
sell.

**Exit logic**: Divergence resolves (OBV and price slopes align) OR 48h timeout.

---

## 8. Implementation Roadmap

### Phase 1: Enhance Existing (1-2 days)

1. **Upgrade VolumeDivergenceStrategy** with multi-timescale Z-scores:
   - Add 1h and 24h Z-score windows alongside existing 50-bar window
   - Add tiered spike classification (2x, 5x, 10x)
   - Add $10,000 minimum volume floor

2. **Upgrade LineMovementStrategy** with ROC divergence:
   - Add ROC slope computation alongside price slope
   - Detect divergence when ROC slope opposes price slope
   - Use as exit signal for momentum trades

3. **Upgrade SmartMoneyDetector** with multi-tier classification:
   - Add retail/informed/whale/institutional tiers
   - Track accumulation patterns (5+ same-direction whale orders in 6h)

### Phase 2: Build New Indicators (3-5 days)

4. **Build ResidualRSI** as new strategy class:
   - Implement RSI on calibration residual
   - Integrate with CalibrationEdgeStrategy for residual computation
   - Register with AlphaCombiner

5. **Build AdaptedMACD** as new strategy class:
   - Implement PM-calibrated MACD (6h/24h/4h)
   - Focus on divergence detection logic
   - Register with AlphaCombiner

6. **Build PMVWAP** as new strategy class:
   - Implement rolling 24h VWAP with deviation bands
   - Mean reversion entry/exit logic
   - Register with AlphaCombiner

7. **Build OBVAdapted** as new strategy class:
   - Implement trade-classified OBV
   - OBV-price divergence detection
   - Register with AlphaCombiner

### Phase 3: Backtest & Tune (3-5 days)

8. **Backtest each indicator individually** on PMXT archive data:
   - Win rate, Sharpe, drawdown per indicator
   - Parameter sensitivity analysis
   - Identify optimal lookback windows

9. **Backtest combinations** using AlphaCombiner:
   - Test the 4 combinations from Section 5.2
   - Measure effective-N improvement from adding indicators
   - Confirm independence from calibration edge

10. **Walk-forward validation**:
    - Use existing `walk_forward.py` infrastructure
    - Verify no overfitting of indicator parameters
    - Compare against Phase 2 baseline (v2_no_maker: 82.4% WR, +301%, Sharpe 3.95)

### Phase 4: Integration & Production (2-3 days)

11. **Update ensemble weights** in production:
    - Let AlphaCombiner derive data-driven weights
    - Phase in new indicators with small weight, increase as IC stabilizes

12. **Add monitoring**:
    - Track per-indicator IC in `monitoring/drift_detector.py`
    - Alert when indicator IC drops below threshold
    - Log signal generation rate per indicator

---

## Appendix A: Parameter Summary Table

| Indicator | Key Parameter | Default | Range to Test |
|---|---|---|---|
| Vol Z-Score (short) | 1h window, 6h baseline | 5x threshold | 2x - 10x |
| Vol Z-Score (long) | 24h window, 7d baseline | 3x threshold | 2x - 5x |
| Residual RSI | Period | 14 bars (1h) | 7 - 21 |
| Residual RSI | Overbought/Oversold | 70/30 | 65-80 / 20-35 |
| MACD | Fast/Slow/Signal | 6h/24h/4h | 4-8h / 18-36h / 3-6h |
| MACD | Divergence window | 6h | 4h - 12h |
| ROC | Short/Med/Long | 6h/24h/72h | 4-12h / 12-48h / 48-168h |
| VWAP | Window | 24h rolling | 12h - 48h |
| VWAP | Entry Z-score | 2.0 | 1.5 - 3.0 |
| OBV | Aggregation | 1h | 30min - 4h |
| OBV | Trend window | 24h | 12h - 48h |
| OBV | Min slope | $500/h | $200 - $1,000/h |

## Appendix B: Expected Independence Structure

Signals should be as independent as possible to maximize effective-N in the
Fundamental Law framework. Expected correlation structure:

```
                CalEdge  VolDiv  OBImb  LinMov  MeanRev  ResRSI  MACD   VWAP   OBV
CalEdge          1.00    0.10   0.05   0.15    0.30     0.40   0.15   0.25   0.10
VolDiv                   1.00   0.35   0.20    0.10     0.10   0.15   0.10   0.30
OBImb                           1.00   0.25    0.10     0.05   0.10   0.05   0.40
LinMov                                 1.00    -0.20    0.30   0.45   0.20   0.15
MeanRev                                        1.00     0.35   0.20   0.50   0.10
ResRSI                                                  1.00   0.35   0.30   0.10
MACD                                                           1.00   0.20   0.15
VWAP                                                                  1.00   0.10
OBV                                                                          1.00
```

Key independence clusters:
1. **Calibration cluster**: CalEdge, ResRSI, MeanRev (correlated ~0.30-0.40)
2. **Volume cluster**: VolDiv, OBImb, OBV (correlated ~0.30-0.40)
3. **Momentum cluster**: LinMov, MACD (correlated ~0.45)
4. **VWAP**: Moderately independent from all (max correlation ~0.50 with MeanRev)

With 9 strategies and average pairwise correlation ~0.20, expected effective-N is
approximately 5-6, giving IR = IC x sqrt(5.5) ~ 2.3x improvement over single-
strategy IC.

## Appendix C: Key Sources

- Polymarket CLOB API documentation: docs.polymarket.com
- PMXT SDK (OHLCV, orderbook, trades): pmxt.dev, github.com/pmxt-dev/pmxt
- PMXT data archive (backtesting): archive.pmxt.dev
- Academic analysis of Polymarket 2024 election: arxiv.org/html/2603.03136v1
- Polymarket volume methodology: paradigm.xyz (double-counting analysis)
- Systematic edges in prediction markets: quantpedia.com
- Smart money detection patterns: Polyscope scanner methodology
- VWAP mean reversion evidence: quantifiedstrategies.com
- Longshot bias calibration: research on 292M trades across 327K binary contracts
