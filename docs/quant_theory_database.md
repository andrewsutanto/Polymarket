# Quantitative Theory & Strategy Database

**Generated:** 2026-04-10
**Repo:** Polymarket Prediction Market Trading Bot
**Purpose:** Comprehensive registry of all theories, models, and strategies under test

---

## Summary Table

| # | Name | Category | Status | Confidence | Backtest WR | Sharpe |
|---|------|----------|--------|------------|-------------|--------|
| 1 | Markov Chain Pricing | Pricing Model | Implemented | Medium | 82.4%* | 3.95* |
| 2 | Longshot Bias Calibration | Pricing Model | Implemented | High | -- | -- |
| 3 | Alpha Combination (Fundamental Law) | Combination | Implemented | High | -- | -- |
| 4 | Bayesian Strategy Updater | Combination | Implemented | High | -- | -- |
| 5 | Combinatorial Arbitrage | Signal | Implemented | High | -- | -- |
| 6 | Smart Money Flow | Signal | Implemented | Medium | -- | -- |
| 7 | Implied Probability Arbitrage | Signal | Implemented | High | -- | -- |
| 8 | Mean Reversion (Post-Update) | Signal | Implemented | Medium | -- | -- |
| 9 | Volume-Price Divergence | Signal | Implemented | Medium-Low | -- | -- |
| 10 | Stale Market Detection | Signal | Implemented | Medium | -- | -- |
| 11 | Line Movement / Momentum | Signal | Implemented | Medium | -- | -- |
| 12 | Cross-Market Consistency Arb | Signal | Implemented | Medium-High | -- | -- |
| 13 | Orderbook Imbalance | Signal | Implemented | Medium | -- | -- |
| 14 | Temporal Bias (3 sub-signals) | Signal | Implemented | Medium | -- | -- |

\* Walk-forward backtest headline results are for the full ensemble (v2_no_maker variant, 200 markets). Individual strategy-level breakdowns not yet isolated.

---

## Detailed Entries

---

### 1. Markov Chain Pricing Model

**File:** `core/markov_model.py`
**Category:** Pricing Model
**Status:** Implemented and wired into ensemble

**Core Hypothesis:**
Prediction market prices follow a discrete Markov process. By building a transition matrix from observed price history and simulating forward paths, we can estimate the true resolution probability independently of the current market price. Deviations between the model estimate and the market price represent exploitable edge.

**Mathematical Formulation:**
- State discretization: s_t = clip(floor(p_t * N), 0, N-1), N=10 states
- Transition matrix: T_ij = count(s_t=i, s_{t+1}=j) / sum_k count(s_t=i, s_{t+1}=k)
- Monte Carlo: P(YES) = (1/M) * sum_{m=1}^{M} 1[s_H^(m) >= s_threshold], M=10,000
- Absorbing state probability via fundamental matrix: N = (I - Q)^{-1}, B = N * R
- Blended estimate: 0.6 * MC_prob + 0.4 * absorbing_prob

**Key Parameters:**
- n_states: 10
- n_simulations: 10,000
- default_horizon_days: 30

**Backtest Result:** Part of ensemble achieving 82.4% WR, +301%, Sharpe 3.95 (200 markets)

**Academic References:**
- Hamilton (1989) -- Markov regime-switching models
- Ang & Bekaert (2002) -- regime shifts in asset allocation

**Confidence:** Medium
- Theoretically sound for modeling price dynamics as a stochastic process
- 10-state discretization is coarse; may lose information in fast-moving markets
- Absorbing state model assumes binary resolution which fits prediction markets well

**Known Weaknesses:**
- Coarse 10-state discretization loses granularity
- Transition matrix needs substantial history (>50 points) for stability
- Non-stationary markets violate the time-homogeneous Markov assumption
- Monte Carlo is slow at 10K paths per market per evaluation cycle
- Blending weights (0.6/0.4 MC/absorbing) are static and not optimized

---

### 2. Longshot Bias Calibration

**File:** `core/bias_calibrator.py`
**Category:** Pricing Model
**Status:** Implemented and wired into Markov model pipeline

**Core Hypothesis:**
Prediction market prices systematically overstate the probability of low-probability events (longshot bias) and understate the probability of high-probability events. YES-side contracts are systematically overpriced by retail participants. Makers earn +1.12% per trade on average; takers lose -1.12%. By calibrating model outputs against empirically observed win rates from 72.1M trades, we correct for these biases.

**Mathematical Formulation:**
- Empirical calibration table: maps market_price_cents -> actual_win_rate (23 data points from 1c to 99c)
- Linear interpolation between nearest calibration points
- NO-side premium adjustment: YES bias -= 1.5% * (0.30 - price) / 0.30 for prices < 30c
- Category multipliers: entertainment 1.8x, sports 1.5x, crypto 1.4x, politics 1.2x, macro 0.8x

**Key Parameters:**
- MAKER_EDGE_PER_TRADE: +1.12%
- TAKER_COST_PER_TRADE: -1.12%
- NO_SIDE_PREMIUM: 1.5%
- Empirical win rate table (23 points)
- Category multipliers (7 categories)

**Backtest Result:** Integrated into all probability estimates; not independently testable

**Academic References:**
- Griffith (1949) -- original longshot bias in horse racing
- Snowberg & Wolfers (2010) -- favorite-longshot bias explanations
- Thaler & Ziemba (1988) -- parimutuel market anomalies
- Becker (2026) -- Polymarket microstructure (72.1M trade analysis, cited in code)

**Confidence:** High
- Based on 72.1M empirical trades from Polymarket itself
- Longshot bias is one of the most replicated findings in prediction market literature
- Category multipliers derived from maker-taker spread data
- NO-side premium is structural (retail overweights YES)

**Known Weaknesses:**
- Calibration table is static; market microstructure may shift over time
- Category multipliers are rough estimates, not rigorously fitted
- Interpolation between sparse calibration points may introduce error at mid-range prices
- Does not account for market maturity (new vs established markets)

---

### 3. Alpha Combination Engine (Fundamental Law)

**File:** `core/alpha_combiner.py`
**Category:** Combination
**Status:** Implemented and central to ensemble

**Core Hypothesis:**
Multiple independent signals with positive Information Coefficients compound into a portfolio-level Information Ratio exceeding any single strategy, per the Fundamental Law of Active Management: IR = IC * sqrt(N_eff). By tracking per-strategy IC via Spearman rank correlation and computing the effective number of independent signals via eigenvalue decomposition, we derive optimal data-driven weights that adapt as the system learns.

**Mathematical Formulation:**
- IR = IC_avg * sqrt(N_eff)
- IC_i = Spearman_rho(predictions_i, outcomes_i) over rolling window
- N_eff = (sum(lambda_j))^2 / sum(lambda_j^2) from eigenvalues of correlation matrix
- w_i^raw = IC_i / sigma_i for IC_i >= IC_min
- w_i = (1 - lambda) * w_i^raw / sum(w_j^raw) + lambda * (1/K), lambda=0.10

**Key Parameters:**
- min_trades_for_ic: 20
- ic_window: 100
- regularization: 0.10
- min_ic_threshold: -0.05

**Backtest Result:** Framework-level; enables the ensemble result

**Academic References:**
- Grinold & Kahn (2000) -- Fundamental Law of Active Management
- Active Portfolio Management (2nd edition)

**Confidence:** High
- Fundamental Law is a cornerstone of institutional quant finance
- Eigenvalue-based N_eff correctly accounts for signal correlation
- Regularization prevents data-mining artifacts from concentrating weights

**Known Weaknesses:**
- IC estimates are noisy with small sample sizes (<50 trades)
- Equal-weight regularization may be suboptimal if prior knowledge exists
- Correlation matrix estimation requires synchronized return series
- Does not account for time-varying IC (IC may decay as a strategy gets crowded)

---

### 4. Bayesian Strategy Weight Updater

**File:** `core/bayesian_updater.py`
**Category:** Combination
**Status:** Implemented, feeds into AlphaCombiner

**Core Hypothesis:**
Strategy performance changes over time due to regime shifts. By maintaining a Beta posterior per strategy for win rates with exponential decay on older observations, the system can rapidly adapt to changing market conditions. When recent IC diverges significantly from historical IC, a regime change is detected and the strategy can be reset or downweighted.

**Mathematical Formulation:**
- Beta conjugate prior: Beta(alpha, beta) for Bernoulli win/loss
- Posterior update: win -> alpha += 1; loss -> beta += 1
- Exponential decay: alpha *= 0.995, beta *= 0.995 per observation (floored at 0.5x prior)
- Posterior mean: E[p] = alpha / (alpha + beta)
- Optimal weight: w_i proportional to E[win_rate_i] * IC_i / sigma_i
- Regime detection: z-score of second-half mean vs first-half mean > 2.0

**Key Parameters:**
- prior_alpha: 1.0 (uniform prior)
- prior_beta: 1.0
- ic_window: 100
- min_observations: 10
- decay_factor: 0.995
- regime_sensitivity: 2.0

**Backtest Result:** Enables adaptive weighting; not independently testable

**Academic References:**
- Chapelle & Li (2011) -- Thompson Sampling
- Ang & Bekaert (2002) -- regime shifts

**Confidence:** High
- Beta-Bernoulli conjugate is the textbook solution for binary outcome learning
- Decay factor elegantly handles non-stationarity
- Regime detection provides a safety valve against structural breaks

**Known Weaknesses:**
- Decay factor (0.995) is static; optimal decay may differ by strategy
- Regime detection is a simple z-test; more sophisticated change-point detection exists
- Reset-to-prior after regime change throws away all learned information
- Does not distinguish between temporary drawdowns and true regime shifts

---

### 5. Combinatorial Arbitrage Engine

**File:** `core/combinatorial_arb.py`
**Category:** Signal (Arbitrage)
**Status:** Implemented with 3-tier detection hierarchy

**Core Hypothesis:**
Multi-outcome prediction markets (NegRisk markets) frequently violate the probability axiom that mutually exclusive, collectively exhaustive outcomes must sum to 1.0. By detecting and trading these violations after accounting for fees, we extract risk-free profit. Historical analysis shows $39.6M was extractable from Polymarket via such violations, with $28.8M from NegRisk rebalancing alone.

**Mathematical Formulation:**
- Constraint: sum(P(outcome_i)) = 1.0 for mutually exclusive outcomes
- Strategy A (buy-all-YES): if sum(yes_prices) < 1.0, buy all YES sides; guaranteed payout = $1
- Strategy B (buy-all-NO): if sum(no_prices) < (n-1), buy all NO sides; payout = $(n-1)
- Rebalancing arb: if yes_price + no_price < 1.0 for single market, buy both
- Net profit = gross_profit - fees * fee_buffer (1.2x safety margin)
- Detection hierarchy: Structural (confidence 0.99) > Tag-based (0.85) > Semantic TF-IDF (0.60)

**Key Parameters:**
- min_profit_pct: 0.5%
- min_link_confidence: 0.60
- max_legs: 6
- fee_buffer: 1.20
- taker_fee_rate: 2% (standard), 5% (crypto)

**Backtest Result:** Not independently backtested yet; theoretical profit is risk-free after fees

**Academic References:**
- arXiv:2508.03474 -- $39.6M extractable value from Polymarket probability violations
- Pearl (2000) -- causal reasoning for market relationship detection

**Confidence:** High
- Arbitrage is the strongest form of alpha (no model risk if constraint is verified)
- 3-tier detection hierarchy ensures high-confidence links are prioritized
- NegRisk structural detection uses contract-enforced mathematical links

**Known Weaknesses:**
- Execution risk: multi-leg trades may partially fill, leaving residual exposure
- Liquidity constraints: many arb opportunities exist in illiquid markets
- Semantic (TF-IDF) clustering at confidence 0.60 may produce false positives
- Fee estimation may be wrong for markets with unusual fee structures
- Latency: other bots may capture arbs before execution completes

---

### 6. Smart Money Flow Detector

**File:** `core/smart_money.py`
**Category:** Signal
**Status:** Implemented

**Core Hypothesis:**
Large orders (>$1,000) on Polymarket are disproportionately placed by informed traders (market makers, quant funds, political insiders). When the net directional pressure from these large orders disagrees with the current market price, the market is likely to move toward the smart money direction. This is a classic informed trading signal adapted from equity markets.

**Mathematical Formulation:**
- Flow imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol), range [-1, 1]
- Exponential time-decay weighting: w = exp(-lambda * age), lambda = ln(2) / half_life
- VWAP computation: vwap_buy = sum(price_i * weighted_size_i) / sum(weighted_size_i)
- Signal trigger: |flow_imbalance| >= 0.4 AND flow direction opposes price lean
- Edge = |imbalance| * |price_lean| * 0.4 + vwap_edge * 0.3

**Key Parameters:**
- large_order_threshold_usd: $1,000
- flow_window_s: 3,600 (1 hour)
- min_large_orders: 3
- flow_disagreement_threshold: 0.40
- decay_half_life_s: 1,800 (30 minutes)
- min_edge: 0.02

**Backtest Result:** Not independently isolated

**Academic References:**
- Kyle (1985) -- insider trading and price impact
- Easley & O'Hara (1987) -- price, trade size, and information
- Barclay & Warner (1993) -- stealth trading
- Easley, Lopez de Prado & O'Hara (2012) -- flow toxicity (VPIN)

**Confidence:** Medium
- Well-grounded in market microstructure theory
- Polymarket's CLOB provides granular trade data needed for this analysis
- Threshold ($1,000) may be too low or too high depending on market

**Known Weaknesses:**
- $1,000 threshold is arbitrary; optimal threshold varies by market liquidity
- Cannot distinguish informed from liquidity-motivated large orders
- VWAP-based edge component is heuristic, not empirically calibrated
- 1-hour flow window may be too long for fast-moving political events
- Spoofing/layering by adversaries could generate false smart money signals

---

### 7. Implied Probability Arbitrage

**File:** `strategies/implied_prob_arb.py`
**Category:** Signal (Structural Arb)
**Status:** Implemented

**Core Hypothesis:**
When a single market's YES + NO prices do not sum to approximately 1.0, one or both sides are structurally mispriced. This requires no external oracle and is the simplest, most reliable strategy. If the sum < 1.0, both sides are underpriced (buy the cheaper one). If the sum > 1.0 + max_vig, both are overpriced (no trade).

**Mathematical Formulation:**
- vig = yes_price + no_price - 1.0
- fair_yes = yes_price / (yes_price + no_price)
- fair_no = no_price / (yes_price + no_price)
- yes_edge = fair_yes - yes_price
- Trade the side with max(yes_edge, no_edge) if edge >= min_edge

**Key Parameters:**
- min_edge: 0.03
- max_vig: 0.04
- min_liquidity: $200
- exit_edge: 0.01

**Backtest Result:** Not independently isolated

**Academic References:**
- Basic no-arbitrage pricing theory
- Page & Clemen (2013) -- prediction market calibration

**Confidence:** High
- Pure structural mispricing with no model risk
- Only requires price observation, no external data
- High win rate expected when sufficient liquidity exists

**Known Weaknesses:**
- Small edge opportunities (typically 3-5%) with limited capacity
- Requires liquidity filter to avoid getting stuck in illiquid markets
- Vig may be intentional market-making spread, not true mispricing
- Edge collapses quickly as other participants notice

---

### 8. Mean Reversion (Post-Update Overshoot)

**File:** `strategies/mean_reversion.py`
**Category:** Signal
**Status:** Implemented

**Core Hypothesis:**
After a NOAA forecast update (or similar information event), Polymarket odds overshoot in the direction of the forecast change, then revert toward fair value as the market digests the information. By fading extreme Z-scores within a post-update time window, we capture the overshoot-reversion cycle.

**Mathematical Formulation:**
- spread = market_price - model_prob
- Z-score = (spread - mean(spread_history)) / std(spread_history)
- Entry: |Z| > 2.0 AND within 30 minutes of forecast update AND historical reversion rate >= 60%
- Exit: |Z| < 0.5 (reverted) OR holding_periods >= 20 (timeout)
- Direction: Z > 2.0 -> SELL (fade overshoot); Z < -2.0 -> BUY

**Key Parameters:**
- zscore_entry: 2.0
- zscore_exit: 0.5
- post_update_window_min: 30
- lookback: 50
- min_reversion_rate: 0.60
- max_holding_periods: 20

**Backtest Result:** Not independently isolated

**Academic References:**
- De Bondt & Thaler (1985) -- market overreaction
- Poterba & Summers (1988) -- mean reversion in stock prices

**Confidence:** Medium
- Overreaction-reversion is a well-documented behavioral finance phenomenon
- Post-update window filter adds specificity
- Reversion rate tracking provides empirical feedback
- Currently tuned for weather markets (NOAA); generalizability unclear

**Known Weaknesses:**
- Highly dependent on the model_prob external oracle (NOAA forecast)
- 30-minute window is arbitrary and may miss slower reversions
- Does not handle cascading updates (second update during hold window)
- Z-score assumes Gaussian distribution of spreads which may not hold
- Lookback of 50 may be too short for robust mean/std estimation

---

### 9. Volume-Price Divergence

**File:** `strategies/volume_divergence.py`
**Category:** Signal
**Status:** Implemented

**Core Hypothesis:**
When volume spikes anomalously high but price barely moves, it indicates informed participants are accumulating positions before the market adjusts. The direction of the small price drift during accumulation predicts the subsequent larger move.

**Mathematical Formulation:**
- volume_zscore = (volume - mean(volume_history)) / std(volume_history)
- Entry: volume_zscore >= 2.5 AND |price_change_5bar| < 0.03
- Direction: follows the sign of the small price drift
- Edge estimate: volume_zscore * 0.01 (heuristic)
- Exit: volume_zscore < 1.0 (volume normalized)

**Key Parameters:**
- volume_zscore_threshold: 2.5
- price_move_threshold: 0.03
- lookback: 50
- min_volume: $100

**Backtest Result:** Not independently isolated

**Academic References:**
- Barclay & Warner (1993) -- stealth trading
- Admati & Pfleiderer (1988) -- intraday volume patterns
- Bouchaud et al. (2018) -- market microstructure

**Confidence:** Medium-Low
- Stealth trading is real in equity markets; less clear in prediction markets
- Edge estimate (zscore * 0.01) is a crude heuristic with no calibration
- Direction inference from small drift is noisy

**Known Weaknesses:**
- Edge formula is entirely heuristic (zscore * 0.01), not empirically fitted
- Small price drift direction is noisy and may be random
- Volume data from Polymarket may not distinguish informed from noise trading
- Lookback of 50 is short for robust volume baseline
- No accounting for volume that's high due to market events (e.g., new listing)

---

### 10. Stale Market Detection

**File:** `strategies/stale_market.py`
**Category:** Signal
**Status:** Implemented

**Core Hypothesis:**
Markets approaching resolution should have prices converging toward 0 or 1. If a market is within 168 hours of resolution and its price hasn't moved meaningfully (range < 5%, std < 2%), it is "stale" and mispriced. Trading toward the nearer boundary captures the eventual convergence.

**Mathematical Formulation:**
- Staleness: price_range < 0.05 AND price_std < 0.02 over lookback window
- Urgency = max(0, 1 - ttl / max_ttl), increases as resolution approaches
- Direction: price > 0.5 -> BUY (converge to 1); price < 0.5 -> SELL (converge to 0)
- Edge = |price - 0.5| * urgency * 0.3
- Exit: if price_range > 0.10 OR price_std > 0.04 (price started moving)

**Key Parameters:**
- stale_hours: 48
- max_ttl_hours: 168 (7 days)
- min_distance_from_boundary: 0.15
- convergence_threshold: 0.85
- min_lookback: 20

**Backtest Result:** Not independently isolated

**Academic References:**
- Shin (1993) -- insider trading and market microstructure
- Leigh, Wolfers & Zitzewitz (2002) -- prediction market efficiency

**Confidence:** Medium
- Convergence near expiry is a structural property of prediction markets
- Urgency weighting is intuitive but the 0.3 scaling factor is uncalibrated
- Stale markets may be stale because there is genuine uncertainty

**Known Weaknesses:**
- Assumes price > 0.5 means YES resolution, which is a coinflip at 0.5
- The 0.3 edge scaling factor is arbitrary
- Cannot distinguish "truly uncertain" from "stale quoting"
- Narrow staleness thresholds (range < 5%, std < 2%) may miss borderline cases
- Does not use any external information to validate convergence direction

---

### 11. Line Movement / Momentum

**File:** `strategies/line_movement.py`
**Category:** Signal
**Status:** Implemented

**Core Hypothesis:**
When market odds are trending in one direction with high R-squared (strong linear fit), the trend has not fully played out. Extrapolating the trend forward provides edge as slow-moving information continues to be priced in.

**Mathematical Formulation:**
- Linear regression on recent min_data_points prices: y = slope * x + intercept
- R-squared = SS_xy^2 / (SS_xx * SS_yy)
- Entry: R^2 >= 0.30 AND |total_move| >= 0.03
- Projected price = intercept + slope * (len + 5)
- Edge = |projected - current_price|
- Direction: slope > 0 -> BUY; slope < 0 -> SELL
- Exit: trend reversal > 0.02 in opposite direction

**Key Parameters:**
- lookback: 100
- trend_threshold: 0.03
- momentum_min_r2: 0.30
- entry_edge: 0.03
- exit_reversal: 0.02
- min_data_points: 20

**Backtest Result:** Not independently isolated

**Academic References:**
- Jegadeesh & Titman (1993) -- momentum returns
- Bouchaud et al. (2018) -- price impact and momentum

**Confidence:** Medium
- Momentum is one of the most robust anomalies in financial markets
- Linear extrapolation is simplistic; real momentum may be nonlinear
- R-squared threshold of 0.30 is low; many noisy series will pass

**Known Weaknesses:**
- Linear regression assumes constant velocity of price change
- R-squared = 0.30 admits significant noise
- 5-bar forward projection is arbitrary
- Trend-following in prediction markets may be less reliable near 0 and 1 boundaries
- No volume confirmation (trend without volume may be spurious)

---

### 12. Cross-Market Consistency Arbitrage

**File:** `strategies/cross_market_arb.py`
**Category:** Signal (Structural Arb)
**Status:** Implemented (depends on relationship_graph module)

**Core Hypothesis:**
Logically related markets sometimes have contradictory pricing. For example, "Trump wins" priced higher than "Republican wins" is a logical impossibility since Trump winning implies the Republican winning. By detecting subset/superset and temporal relationships between markets, we can trade the mispriced side.

**Mathematical Formulation:**
- Subset constraint: P(A) <= P(B) when A implies B
- Temporal constraint: P(by_later_date) >= P(by_earlier_date)
- Gap = violation magnitude between related markets
- Entry: gap >= 0.05 AND relationship_confidence >= 0.50
- Exit: gap <= 0.02

**Key Parameters:**
- min_gap: 0.05
- min_confidence: 0.50
- exit_gap: 0.02

**Backtest Result:** Not independently isolated

**Academic References:**
- Pearl (2000) -- causal reasoning and logical constraints
- arXiv:2508.03474 -- multi-market constraint violations on Polymarket

**Confidence:** Medium-High
- Logical constraints are mathematically provable (no model risk when link is certain)
- Depends heavily on accuracy of the relationship graph detection
- Confidence threshold of 0.50 is relatively permissive

**Known Weaknesses:**
- Relationship detection (from relationship_graph module) may produce false links
- Confidence threshold of 0.50 allows uncertain relationships
- Execution requires trading both sides, with partial fill risk
- Gap may reflect informed disagreement rather than mispricing

---

### 13. Orderbook Imbalance

**File:** `strategies/orderbook_imbalance.py`
**Category:** Signal (Microstructure)
**Status:** Implemented

**Core Hypothesis:**
When bid-side volume significantly outweighs ask-side volume (or vice versa) in the CLOB orderbook, this signals short-term directional pressure before the price adjusts. The imbalance is most predictive within the first few minutes and decays exponentially as the market absorbs the information.

**Mathematical Formulation:**
- Raw ratio = bid_volume_usd / ask_volume_usd (aggregated over top N depth levels)
- Decay-weighted ratio: w = exp(-lambda * age_s), lambda = ln(2) / half_life
- weighted_ratio = sum(w_i * ratio_i) / sum(w_i)
- Entry: weighted_ratio >= 3.0 (BUY) or <= 1/3.0 (SELL)
- Edge = 0.02 * ln(imbalance_ratio) * min(room_to_move * 2, 1.0)
- Strength saturates at ratio = 8.0

**Key Parameters:**
- imbalance_threshold: 3.0
- decay_half_life_s: 180 (3 minutes)
- min_depth_usd: $100
- lookback_window: 50 snapshots
- min_snapshots: 3
- strength_cap_ratio: 8.0
- depth_levels: 5

**Backtest Result:** Not independently isolated

**Academic References:**
- Cont, Kukanov & Stoikov (2014) -- orderbook event price impact
- Cartea, Jaimungal & Penalva (2015) -- high-frequency trading
- Bouchaud et al. (2018) -- orderbook microstructure

**Confidence:** Medium
- Orderbook imbalance is a well-studied microstructure signal
- Exponential decay is a principled way to handle signal staleness
- 3:1 threshold and logarithmic edge scaling are reasonable starting points

**Known Weaknesses:**
- Orderbook data from Polymarket CLOB may be thin (few levels)
- 3-minute decay half-life may be too fast or too slow
- Spoofing (fake orders pulled before execution) creates false signals
- Edge formula (0.02 * ln(ratio)) is heuristic, not calibrated
- Only uses top 5 levels; deeper book information is ignored
- $100 minimum depth is very low and may trigger on noise

---

### 14. Temporal Bias (3 Sub-Signals)

**File:** `strategies/temporal_bias.py`
**Category:** Signal (Structural)
**Status:** Implemented

**Core Hypothesis:**
Prediction markets exhibit persistent time-dependent structural inefficiencies arising from the composition of the participant base: (a) near-expiry markets converge faster, so stale prices near expiry are high-conviction mispricings; (b) Asian-session hours have ~60% wider spreads due to low US participation; (c) weekend price drift tends to revert on Monday.

**Sub-Signal 14a: Expiry Convergence**
- urgency = exp(-0.15 * hours_to_resolution)
- edge = urgency * distance_from_edge * 0.15 + spread * 0.5, boosted by 1.5x
- Direction: price >= 0.5 -> BUY; price < 0.5 -> SELL

**Sub-Signal 14b: Off-Session Spread Capture**
- Session multipliers: Asian 1.60, European 1.20, US 1.00, Off-hours 1.35
- Weekend multiplier: 1.40
- Edge = excess_spread * 0.5 (capture half the spread widening)
- Trade: buy the cheaper side (act as maker)

**Sub-Signal 14c: Weekend Drift Reversion**
- Monday before US session only
- drift = current_price - friday_close
- Edge = |drift| * 0.6 (expect 60% mean reversion)
- Direction: opposes the drift

**Key Parameters:**
- min_edge: 0.02
- expiry_hours_threshold: 24
- expiry_edge_boost: 1.5
- spread_ratio_threshold: 1.5
- weekend_drift_threshold: 0.03
- min_liquidity: $200
- off_session_min_spread: 0.03

**Backtest Result:** Not independently isolated

**Academic References:**
- Harris (1986) -- intraday patterns in stock returns
- Keim & Stambaugh (1984) -- weekend effect
- Admati & Pfleiderer (1988) -- intraday volume/price patterns

**Confidence:** Medium
- Time-of-day and weekend effects are well-documented in equity markets
- Prediction market-specific session multipliers need empirical validation
- Expiry convergence is structurally sound
- 60% weekend reversion assumption is uncalibrated

**Known Weaknesses:**
- Session spread multipliers (1.60, 1.20, etc.) are estimates, not fitted to data
- Weekend reversion rate (60%) is assumed, not measured
- Expiry convergence direction (price > 0.5 = YES) is a heuristic with ~50% accuracy at midrange
- Requires friday_close reference price which may not always be available
- Off-session spread capture requires maker order placement (execution complexity)

---

## Strategy Gap Analysis: Missing Theories

The following are theories and approaches from quantitative finance and prediction market literature that are NOT currently implemented but could add meaningful alpha:

### Gap 1: Sentiment Analysis / NLP Signals
**Theory:** News articles, social media, and Polymarket comment sections contain leading indicators of price movement. NLP models can extract sentiment and positioning signals from text data.
**Academic basis:** Tetlock (2007) -- media sentiment predicts stock returns
**Priority:** High -- Polymarket is heavily driven by news events; NLP could front-run slow price adjustment.
**Difficulty:** High -- requires real-time text ingestion and NLP pipeline.

### Gap 2: VPIN (Volume-Synchronized Probability of Informed Trading)
**Theory:** Classifies trade flow as informed vs uninformed using volume buckets rather than time buckets. More robust than simple smart money detection.
**Academic basis:** Easley, Lopez de Prado & O'Hara (2012)
**Priority:** High -- directly improves the smart money signal.
**Difficulty:** Medium -- requires adapting VPIN to prediction market trade data.

### Gap 3: Market Making Strategy
**Theory:** Systematically provide liquidity by quoting both sides and earning the bid-ask spread. Use inventory management and adverse selection models to set quotes optimally.
**Academic basis:** Cartea, Jaimungal & Penalva (2015); Avellaneda & Stoikov (2008)
**Priority:** High -- the bias calibrator shows makers earn +1.12% per trade structurally.
**Difficulty:** High -- requires real-time quoting infrastructure and inventory management.

### Gap 4: Kelly Criterion Position Sizing Optimization
**Theory:** While Quarter-Kelly is mentioned in the architecture, there is no dedicated module that dynamically optimizes the Kelly fraction based on current edge certainty, correlation structure, and account constraints.
**Academic basis:** Thorp (2006) -- Kelly criterion in betting and finance
**Priority:** Medium -- could significantly improve risk-adjusted returns.
**Difficulty:** Medium -- requires robust edge and variance estimation per trade.

### Gap 5: Hawkes Process / Event Clustering
**Theory:** Trades arrive in clusters (self-exciting). A Hawkes process model can identify when activity is endogenously driven (herding) vs exogenously driven (new information). This improves signal filtering.
**Academic basis:** Bacry et al. (2015) -- Hawkes processes in finance
**Priority:** Medium -- would improve the smart money and volume divergence signals.
**Difficulty:** Medium-High -- Hawkes process estimation requires specialized fitting.

### Gap 6: Transfer Learning from Sports Betting Markets
**Theory:** Polymarket political/event markets share structural similarities with sports betting markets (binary outcomes, known resolution times, public information). Feature engineering and model selection insights from sports betting research could be transferred.
**Academic basis:** Shin (1993); Rothschild & Sethi (2015)
**Priority:** Medium -- rich prior literature to mine.
**Difficulty:** Low -- primarily a research and feature engineering task.

### Gap 7: Partisan/Tribal Bias Modeling
**Theory:** Political prediction markets exhibit systematic partisan bias where traders overweight their preferred candidate. A model that identifies and quantifies this bias by market category and event type could provide consistent alpha.
**Academic basis:** Rothschild (2009) -- bias in political prediction markets; Rothschild & Sethi (2015) -- wishful thinking
**Priority:** Medium-High -- Polymarket is dominated by political markets.
**Difficulty:** Medium -- requires partisan proxy data (polling, demographics).

### Gap 8: Liquidity-Adjusted Fair Value
**Theory:** Current fair value models do not adjust for illiquidity premium. Thin markets should trade at a discount to fundamental value because holders bear exit risk. A model that estimates the liquidity discount and trades markets transitioning from illiquid to liquid could capture spread compression.
**Academic basis:** Amihud (2002) -- illiquidity and stock returns
**Priority:** Medium -- structural and persistent.
**Difficulty:** Low-Medium -- liquidity data is available from the CLOB.

### Gap 9: Correlated Multi-Market Factor Model
**Theory:** Build a factor model across all active Polymarket markets to identify common risk factors (e.g., "Trump factor", "crypto sentiment factor", "AI regulation factor"). Markets that diverge from their factor-implied fair value are mispriced.
**Academic basis:** Fama & French (1993) -- multi-factor models; Ross (1976) -- APT
**Priority:** Medium -- would improve cross-market signals and portfolio construction.
**Difficulty:** High -- requires defining prediction market factors (non-trivial).

### Gap 10: Adversarial Detection / Bot Fingerprinting
**Theory:** Identify competing trading bots by their order patterns (fixed sizes, periodic timing, queue position behavior). Knowing when a competitor bot is active allows us to avoid adverse selection or exploit their predictable behavior.
**Academic basis:** Cont, Kukanov & Stoikov (2014)
**Priority:** Low-Medium -- defensive improvement.
**Difficulty:** High -- requires pattern recognition on trade flow.

---

## Cross-Reference: Theory-to-File Map

| Theory | Primary File | Supporting Files |
|--------|-------------|-----------------|
| Markov Chain Pricing | core/markov_model.py | core/bias_calibrator.py |
| Longshot Bias | core/bias_calibrator.py | -- |
| Fundamental Law | core/alpha_combiner.py | core/bayesian_updater.py |
| Bayesian Updating | core/bayesian_updater.py | core/alpha_combiner.py |
| Combinatorial Arb | core/combinatorial_arb.py | -- |
| Smart Money Flow | core/smart_money.py | -- |
| Implied Prob Arb | strategies/implied_prob_arb.py | core/fair_value.py |
| Mean Reversion | strategies/mean_reversion.py | -- |
| Volume Divergence | strategies/volume_divergence.py | -- |
| Stale Market | strategies/stale_market.py | -- |
| Line Movement | strategies/line_movement.py | -- |
| Cross-Market Arb | strategies/cross_market_arb.py | core/relationship_graph.py |
| Orderbook Imbalance | strategies/orderbook_imbalance.py | -- |
| Temporal Bias | strategies/temporal_bias.py | -- |
