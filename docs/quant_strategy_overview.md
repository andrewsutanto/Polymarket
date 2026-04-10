# Quantitative Strategy Overview: Polymarket Prediction Market Trading System

**Version:** 1.0
**Date:** 2026-04-10
**Classification:** Internal / Reviewer Copy

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Signal Combination Framework](#3-signal-combination-framework)
4. [Core Models](#4-core-models)
   - 4.1 [Markov Chain Pricing Model](#41-markov-chain-pricing-model)
   - 4.2 [Longshot Bias Calibrator](#42-longshot-bias-calibrator)
   - 4.3 [Combinatorial Arbitrage Engine](#43-combinatorial-arbitrage-engine)
   - 4.4 [Bayesian Strategy Weight Updater](#44-bayesian-strategy-weight-updater)
   - 4.5 [Smart Money Flow Detector](#45-smart-money-flow-detector)
5. [Trading Strategies](#5-trading-strategies)
   - 5.1 [Implied Probability Arbitrage](#51-implied-probability-arbitrage)
   - 5.2 [Mean Reversion](#52-mean-reversion)
   - 5.3 [Volume-Price Divergence](#53-volume-price-divergence)
   - 5.4 [Stale Market Detection](#54-stale-market-detection)
   - 5.5 [Line Movement / Momentum](#55-line-movement--momentum)
   - 5.6 [Cross-Market Consistency Arbitrage](#56-cross-market-consistency-arbitrage)
   - 5.7 [Orderbook Imbalance](#57-orderbook-imbalance)
   - 5.8 [Temporal Bias](#58-temporal-bias)
6. [Portfolio Construction](#6-portfolio-construction)
7. [Backtesting Methodology](#7-backtesting-methodology)
8. [Simulated Environment Architecture](#8-simulated-environment-architecture)
9. [Known Risks and Overfitting Assessment](#9-known-risks-and-overfitting-assessment)
10. [Parameter Registry](#10-parameter-registry)
11. [References](#11-references)

---

## 1. Executive Summary

This document describes the full quantitative framework for a systematic trading system operating on Polymarket, a blockchain-based prediction market exchange. The system combines eight alpha-generating strategies with four core analytical models under an institutional-grade signal combination framework grounded in the Fundamental Law of Active Management.

**Core thesis:** Prediction markets exhibit persistent structural inefficiencies arising from (a) retail participant biases (longshot bias, YES-side bias, emotional betting), (b) microstructure effects (time-of-day spreads, stale quoting near expiry), and (c) multi-market constraint violations (probability axiom breaches across related outcomes). These inefficiencies are well-documented in academic literature and are exploitable by a disciplined, model-driven approach.

**Key architectural principles:**

- **Signal combination via the Fundamental Law:** IR = IC x sqrt(N), where N independent signals with positive IC compound into a portfolio-level information ratio exceeding any single strategy.
- **Bayesian weight adaptation:** Strategy allocations update continuously via conjugate Beta posteriors, adapting to regime changes.
- **Bias calibration from empirical data:** All probability estimates pass through a longshot bias corrector trained on 72.1 million historical Polymarket trades.
- **True out-of-sample validation:** The simulation environment enforces strict temporal isolation -- the trading bot receives only data available up to the current simulated timestamp, with fills executed at actual book prices after real computation latency.

**Walk-forward backtest headline results (200 markets):** 82.4% win rate, +301% cumulative return, Sharpe ratio 3.95 (v2_no_maker variant).

---

## 2. System Architecture

The system is composed of three layers:

```
+-----------------------------------------------------------------------+
|                         SIGNAL LAYER                                   |
|  8 strategies + smart money detector produce independent signals       |
+-----------------------------------------------------------------------+
         |                                                    |
         v                                                    v
+----------------------------+    +------------------------------+
|   ALPHA COMBINER           |    |   BAYESIAN UPDATER           |
|   IR = IC x sqrt(N)        |--->|   Beta(alpha, beta) priors   |
|   Optimal weights via IC/σ |<---|   Regime change detection    |
+----------------------------+    +------------------------------+
         |
         v
+-----------------------------------------------------------------------+
|                      PORTFOLIO LAYER                                   |
|  Quarter-Kelly sizing | Position limits | Drawdown controls            |
+-----------------------------------------------------------------------+
         |
         v
+-----------------------------------------------------------------------+
|                      EXECUTION LAYER                                   |
|  CLOB API | Fill at best ask/bid | Slippage tracking | Fee accounting  |
+-----------------------------------------------------------------------+
```

Each strategy implements the `BaseStrategy` interface:

```
generate_signal(market_data) -> Signal | None
```

Where `Signal` carries: direction (BUY/SELL), strength [0,1], edge estimate, strategy name, market identifiers, and metadata.

---

## 3. Signal Combination Framework

### 3.1 The Fundamental Law of Active Management

Following Grinold and Kahn (2000), the portfolio's expected information ratio is:

$$
IR = \overline{IC} \times \sqrt{N_{\text{eff}}}
$$

Where:
- $\overline{IC}$ = mean Information Coefficient across strategies (Spearman rank correlation of predicted edge vs. realized return)
- $N_{\text{eff}}$ = effective number of independent signals (not raw count)

### 3.2 Information Coefficient Computation

For each strategy $i$, IC is computed as Spearman rank correlation over a rolling window:

$$
IC_i = \rho_s(\hat{r}_{i,1}, \hat{r}_{i,2}, \ldots, \hat{r}_{i,n}; r_1, r_2, \ldots, r_n)
$$

Where $\hat{r}_{i,k}$ is the signed predicted edge for trade $k$ and $r_k$ is the realized return. Minimum 20 observations required before IC is trusted.

### 3.3 Effective N via Eigenvalue Decomposition

Raw signal count overcounts when strategies are correlated. The effective number of independent bets is computed from the eigenvalues of the strategy return correlation matrix:

$$
N_{\text{eff}} = \frac{\left(\sum_{j=1}^{K} \lambda_j\right)^2}{\sum_{j=1}^{K} \lambda_j^2}
$$

Where $\lambda_j$ are eigenvalues of the $K \times K$ correlation matrix of strategy returns. When all strategies are perfectly independent, $N_{\text{eff}} = K$. When perfectly correlated, $N_{\text{eff}} = 1$.

### 3.4 Optimal Weight Computation

Strategy weights are proportional to IC divided by return volatility, with regularization toward equal weights:

$$
w_i^{\text{raw}} = \frac{IC_i}{\sigma_i}, \quad IC_i \geq IC_{\min}
$$

$$
w_i = (1 - \lambda) \cdot \frac{w_i^{\text{raw}}}{\sum_j w_j^{\text{raw}}} + \lambda \cdot \frac{1}{K}
$$

Where $\lambda = 0.10$ (regularization) prevents concentration in low-data strategies.

Strategies with $IC_i < IC_{\min} = -0.05$ receive zero weight (effectively disabled).

### 3.5 Combined Signal

The final trading signal aggregates across strategies:

$$
\text{edge}_{\text{combined}} = \frac{\sum_{i \in S} w_i \cdot \text{edge}_i \cdot \text{strength}_i}{\sum_{i \in S} w_i}
$$

Where $S$ is the set of strategies agreeing on direction (BUY or SELL). The direction with the larger weighted edge wins.

**Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `min_trades_for_ic` | 20 | Minimum trades before IC computation |
| `ic_window` | 100 | Rolling window for IC |
| `regularization` | 0.10 | Shrinkage toward equal weights |
| `min_ic_threshold` | -0.05 | Below this, strategy gets zero weight |

---

## 4. Core Models

### 4.1 Markov Chain Pricing Model

**Source:** `core/markov_model.py`

#### 4.1.1 Mathematical Formulation

The model discretizes the [0, 1] price space into $N$ states and constructs a transition matrix from observed price history.

**State discretization:**

$$
s_t = \text{clip}\left(\lfloor p_t \times N \rfloor, 0, N-1\right)
$$

Where $p_t \in [0.001, 0.999]$ is the market price at time $t$ and $N = 10$ states.

**Transition matrix construction:**

$$
T_{ij} = \frac{\text{count}(s_t = i, s_{t+1} = j)}{\sum_{k=0}^{N-1} \text{count}(s_t = i, s_{t+1} = k)}
$$

Rows with zero observations default to uniform: $T_{ij} = 1/N$.

#### 4.1.2 Monte Carlo Probability Estimation

Starting from $s_0 = \lfloor p_{\text{current}} \times N \rfloor$, simulate 10,000 paths of length $H$ (horizon) steps:

$$
\hat{P}(\text{YES}) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\left[s_H^{(m)} \geq s_{\text{threshold}}\right]
$$

Where $M = 10{,}000$, $s_{\text{threshold}} = \lfloor 0.5 \times N \rfloor$, and each path samples from the transition matrix:

$$
s_{t+1}^{(m)} \sim \text{Categorical}(T[s_t^{(m)}, :])
$$

#### 4.1.3 Absorbing State Analysis

The model also treats states 0 and $N-1$ as absorbing barriers (NO and YES resolution). For transient states $\{1, \ldots, N-2\}$:

$$
Q = T[1:N-1, 1:N-1] \quad (\text{transient-to-transient})
$$

$$
R = \begin{bmatrix} T[1:N-1, 0] & T[1:N-1, N-1] \end{bmatrix} \quad (\text{transient-to-absorbing})
$$

The fundamental matrix and absorption probabilities:

$$
\mathcal{N} = (I - Q)^{-1}, \quad B = \mathcal{N} R
$$

$$
P(\text{absorbed into YES} | s_0) = B[s_0 - 1, 1]
$$

#### 4.1.4 Blended Estimate

The final raw probability blends MC and absorbing estimates:

$$
\hat{p}_{\text{raw}} = 0.6 \cdot \hat{p}_{\text{MC}} + 0.4 \cdot \hat{p}_{\text{absorb}}
$$

The MC estimate is weighted more heavily because it is more reliable with sufficient data.

#### 4.1.5 Steady-State Distribution

Computed via eigendecomposition of $T^{\top}$:

$$
T^{\top} \pi = \pi, \quad \pi_i \geq 0, \quad \sum_i \pi_i = 1
$$

The steady state indicates the long-run equilibrium price distribution.

#### 4.1.6 Confidence Score

$$
\text{confidence} = \min\left(\frac{|\text{history}|}{50}, 1.0\right) \times 0.7 + \left(1 - |\hat{p}_{\text{MC}} - \hat{p}_{\text{absorb}}|\right) \times 0.3
$$

Agreement between MC and absorbing methods boosts confidence.

#### 4.1.7 Edge Calculation

$$
\text{edge} = \hat{p}_{\text{calibrated}} - p_{\text{market}}
$$

Where $\hat{p}_{\text{calibrated}}$ is the output of the bias calibrator (Section 4.2).

**Parameters:**

| Parameter | Value |
|---|---|
| `n_states` | 10 |
| `n_simulations` | 10,000 (backtest: 2,000 for speed) |
| `default_horizon_days` | 30 |
| Minimum history for estimation | 5 price points |

**Assumptions and Limitations:**
- Markov property (memoryless transitions) is a simplification; real prediction market prices exhibit momentum and clustering.
- 10-state discretization is coarse; finer grids would require more data for reliable transition estimates.
- Horizon choice of 30 steps is arbitrary and should be calibrated to median time-to-resolution.

**Academic context:** Markov chain models for financial time series are well-established (Hamilton, 1989; Ang and Bekaert, 2002). Application to prediction markets follows the same logic as regime-switching models in bond pricing.

---

### 4.2 Longshot Bias Calibrator

**Source:** `core/bias_calibrator.py`

#### 4.2.1 Empirical Foundation

The calibrator is trained on an empirical study of 72.1 million Polymarket trades. The key finding: cheap contracts are systematically overpriced. A 1-cent YES contract actually wins only 0.43% of the time (not 1%), a 5-cent contract wins 4.18% (not 5%), and NO outperforms YES at 69 of 99 price levels.

This is the prediction market analogue of the well-documented longshot bias in horse racing (Griffith, 1949; Thaler and Ziemba, 1988) and sports betting (Snowberg and Wolfers, 2010).

#### 4.2.2 Calibration Function

The calibrator maps naive probability $p$ to empirically observed win rate $\hat{w}(p)$ via linear interpolation over the empirical table:

$$
\hat{w}(p) = w_{\text{lo}} + \frac{p \times 100 - c_{\text{lo}}}{c_{\text{hi}} - c_{\text{lo}}} \cdot (w_{\text{hi}} - w_{\text{lo}})
$$

Where $c_{\text{lo}}, c_{\text{hi}}$ are the nearest empirical price-cent data points and $w_{\text{lo}}, w_{\text{hi}}$ are their observed win rates.

Selected calibration points:

| Market Price (cents) | Naive Probability | Empirical Win Rate | Bias (naive - actual) |
|---|---|---|---|
| 1 | 1.00% | 0.43% | +0.57pp |
| 5 | 5.00% | 4.18% | +0.82pp |
| 10 | 10.00% | 8.92% | +1.08pp |
| 20 | 20.00% | 18.10% | +1.90pp |
| 50 | 50.00% | 49.50% | +0.50pp |
| 80 | 80.00% | 82.80% | -2.80pp |
| 95 | 95.00% | 95.80% | -0.80pp |

At low prices, YES is overpriced (longshot bias). At high prices, YES is slightly underpriced (favorite-longshot effect reversal).

#### 4.2.3 NO-Side Premium Adjustment

YES is systematically overpriced by retail flow (69/99 price levels). For YES-side estimates:

$$
\hat{p}_{\text{adj}} = \hat{p} - \frac{0.015 \times (0.30 - p_{\text{market}})}{0.30}, \quad p_{\text{market}} < 0.30
$$

$$
\hat{p}_{\text{adj}} = \hat{p} + \frac{0.015 \times (p_{\text{market}} - 0.70)}{0.30}, \quad p_{\text{market}} > 0.70
$$

The 1.5% NO-side premium is largest at extreme low prices (up to 57% NO edge at 1 cent) and reverses slightly at high prices.

#### 4.2.4 Category Edge Multipliers

Different market categories exhibit different levels of inefficiency, driven by participant composition:

| Category | Multiplier | Rationale |
|---|---|---|
| Entertainment | 1.80 | 7.32pp maker-taker gap; retail-dominated |
| Sports | 1.50 | Emotional betting, tribal bias |
| Crypto | 1.40 | High volatility, narrative-driven |
| Politics | 1.20 | Tribal YES bias on favored candidates |
| Science | 1.10 | Moderate inefficiency |
| Macro | 0.80 | Near-efficient; quant trader presence |
| Other | 1.00 | Baseline |

These multipliers scale position sizing, not the probability estimate directly.

#### 4.2.5 Maker-Taker Edge

From the 72.1M trade analysis:

$$
\text{Maker edge per trade} = +1.12\%
$$

$$
\text{Taker cost per trade} = -1.12\%
$$

The system operates as a maker (limit orders) whenever possible to capture this structural advantage.

**Assumptions and Limitations:**
- Calibration data is historical and may not persist as the market matures.
- Category multipliers are based on aggregate statistics; individual markets within a category vary widely.
- The NO-side premium is an average effect and may not hold for all market structures.

---

### 4.3 Combinatorial Arbitrage Engine

**Source:** `core/combinatorial_arb.py`

#### 4.3.1 Theoretical Basis

For a set of mutually exclusive, collectively exhaustive (MECE) outcomes $\{O_1, O_2, \ldots, O_K\}$, probability theory requires:

$$
\sum_{k=1}^{K} P(O_k) = 1
$$

If market prices violate this constraint:

$$
\sum_{k=1}^{K} p_k \neq 1
$$

Then a riskless arbitrage exists. This is documented extensively in Polymarket specifically by arxiv:2508.03474, which found $39.6M in extractable value, with $28.8M from NegRisk market rebalancing.

#### 4.3.2 Detection Hierarchy

The engine uses a three-tier detection scheme:

**Tier 1 -- STRUCTURAL (confidence: 0.95-0.99):** Markets sharing `event_slug` or `negRiskMarketID` are mathematically linked by contract design. The API groups these natively. NegRisk markets are the richest source of arbitrage.

**Tier 2 -- TAG-BASED (confidence: 0.85):** Markets sharing `groupItemTitle` or significant tag overlap. High confidence but requires manual verification.

**Tier 3 -- SEMANTIC (confidence: 0.60):** TF-IDF cosine similarity fallback for uncategorized markets.

$$
\text{sim}(d_i, d_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \cdot \|\vec{v}_j\|}
$$

Where $\vec{v}_i$ is the TF-IDF vector of market question $i$, with IDF:

$$
\text{IDF}(w) = \ln\frac{N}{1 + \text{df}(w)}
$$

Similarity threshold: 0.45. Only markets in the same category are compared.

#### 4.3.3 Arbitrage Strategies

**Strategy A -- Buy All YES:**

If $\sum_{k=1}^{K} p_k^{\text{YES}} < 1$, buying all YES tokens costs $\sum p_k^{\text{YES}}$ and pays exactly \$1 at settlement.

$$
\text{Gross profit} = 1 - \sum_{k=1}^{K} p_k^{\text{YES}}
$$

$$
\text{Net profit} = \text{Gross} - \sum_{k=1}^{K} p_k^{\text{YES}} \times f \times b
$$

Where $f$ is the fee rate (2% standard, 5% crypto) and $b = 1.2$ is the fee safety buffer.

**Strategy B -- Buy All NO:**

If $\sum_{k=1}^{K} p_k^{\text{NO}} < K - 1$, buying all NO tokens costs $\sum p_k^{\text{NO}}$ and pays exactly \$(K-1) at settlement (since exactly one YES resolves, making $K-1$ NO tokens winners).

$$
\text{Gross profit} = (K - 1) - \sum_{k=1}^{K} p_k^{\text{NO}}
$$

**Single-Market Rebalancing:**

For binary markets where $p_{\text{YES}} + p_{\text{NO}} < 1$:

$$
\text{Cost} = p_{\text{YES}} + p_{\text{NO}} < 1
$$

$$
\text{Guaranteed payout} = 1.00
$$

#### 4.3.4 Minimum Profitability Filter

$$
\text{ROI} = \frac{\text{Net profit}}{\text{Total cost}} \times 100 \geq 0.5\%
$$

Only opportunities with ROI above 0.5% after fees (with 20% buffer) are emitted.

**Parameters:**

| Parameter | Value |
|---|---|
| `min_profit_pct` | 0.5% |
| `min_link_confidence` | 0.60 |
| `max_legs` | 6 |
| `fee_buffer` | 1.20 (20% safety margin) |
| `taker_fee_rate` | 2% (standard) |
| `crypto_fee_rate` | 5% (crypto markets) |
| Cosine similarity threshold | 0.45 |

**Assumptions and Limitations:**
- Structural links (Tier 1) are mathematically guaranteed. Semantic links (Tier 3) are approximate and may produce false clusters.
- Execution risk: all legs must fill to capture the arbitrage. Partial fills create directional exposure.
- Leg count is capped at 6 to limit execution complexity.

**Academic context:** Multi-market arbitrage in prediction markets is documented in Leigh et al. (2002) and more recently in the Polymarket-specific study (arXiv:2508.03474).

---

### 4.4 Bayesian Strategy Weight Updater

**Source:** `core/bayesian_updater.py`

#### 4.4.1 Conjugate Beta Prior

Each strategy $i$ maintains a Beta distribution posterior for its win rate:

$$
p_i \sim \text{Beta}(\alpha_i, \beta_i)
$$

Starting from an uninformative prior $\text{Beta}(1, 1)$ (uniform on [0, 1]).

**Posterior update after trade $k$:**

$$
\alpha_i \leftarrow \alpha_i + \mathbf{1}[\text{trade } k \text{ won}]
$$

$$
\beta_i \leftarrow \beta_i + \mathbf{1}[\text{trade } k \text{ lost}]
$$

**Posterior mean (expected win rate):**

$$
\hat{p}_i = \frac{\alpha_i}{\alpha_i + \beta_i}
$$

**Posterior variance:**

$$
\text{Var}(p_i) = \frac{\alpha_i \beta_i}{(\alpha_i + \beta_i)^2 (\alpha_i + \beta_i + 1)}
$$

**95% credible interval (normal approximation):**

$$
CI_{95} = \hat{p}_i \pm 1.96 \sqrt{\text{Var}(p_i)}
$$

#### 4.4.2 Exponential Decay for Regime Responsiveness

To prevent ancient observations from dominating the posterior (making it unresponsive to regime changes), a decay factor $\delta = 0.995$ is applied before each update:

$$
\alpha_i \leftarrow \max(\alpha_i \times \delta, \alpha_0 \times 0.5)
$$

$$
\beta_i \leftarrow \max(\beta_i \times \delta, \beta_0 \times 0.5)
$$

The floor at $0.5 \times$ prior prevents the effective sample size from collapsing entirely.

Effective half-life in observations: $\frac{\ln 2}{\ln(1/\delta)} \approx 138$ trades.

#### 4.4.3 Posterior-Optimal Weights

Strategy weight is proportional to a composite score combining Bayesian win rate, IC, and inverse volatility:

$$
w_i^{\text{raw}} = \frac{\hat{p}_i \times (0.3 + 0.7 \times \max(IC_i, 0))}{\sigma_i}
$$

This is analogous to an expected Sharpe ratio:
- $\hat{p}_i$: Bayesian estimate of win probability
- $IC_i$: rolling Spearman rank correlation (signal quality)
- $\sigma_i$: standard deviation of returns (consistency)

Strategies with fewer than `min_observations` trades receive $w_i = \hat{p}_i \times 0.1$ (heavily discounted prior).

#### 4.4.4 Regime Change Detection

A regime change is flagged when the second half of recent returns diverges from the first half by more than $\gamma$ standard deviations:

$$
z = \frac{|\bar{r}_{\text{second half}} - \bar{r}_{\text{first half}}|}{\sigma_{\text{first half}}}
$$

If $z > \gamma = 2.0$, the strategy is flagged for potential posterior reset.

**Parameters:**

| Parameter | Value |
|---|---|
| `prior_alpha` | 1.0 |
| `prior_beta` | 1.0 |
| `ic_window` | 100 |
| `min_observations` | 10 |
| `decay_factor` | 0.995 |
| `regime_sensitivity` | 2.0 (standard deviations) |

**Assumptions and Limitations:**
- The Beta conjugate model treats each trade as an independent Bernoulli trial, ignoring correlation between consecutive trades.
- Decay factor creates a recency bias that may overreact to short losing streaks.
- Normal approximation for credible intervals is poor when $\alpha$ or $\beta$ is small.

**Academic context:** Bayesian updating for strategy allocation follows Thompson Sampling literature (Chapelle and Li, 2011) and is standard in multi-armed bandit frameworks for portfolio selection.

---

### 4.5 Smart Money Flow Detector

**Source:** `core/smart_money.py`

#### 4.5.1 Core Hypothesis

Large orders on Polymarket are disproportionately placed by informed participants (market makers, quant funds, political insiders). When net large-order flow disagrees with the current market price, this signals private information that has not yet been impounded.

#### 4.5.2 Large Order Classification

A trade is classified as "large" (smart money) if:

$$
\text{size}_{\text{USD}} \geq \tau = \$1{,}000
$$

#### 4.5.3 Time-Decayed Flow Aggregation

Within a lookback window of $W = 3{,}600$ seconds, each order is weighted by exponential time decay:

$$
w_j = \exp\left(-\frac{\ln 2}{\tau_{1/2}} \cdot (t_{\text{now}} - t_j)\right)
$$

Where $\tau_{1/2} = 1{,}800$s (30-minute half-life).

Aggregated flow:

$$
V_{\text{buy}} = \sum_{j \in \text{buys}} w_j \cdot \text{size}_j, \quad V_{\text{sell}} = \sum_{j \in \text{sells}} w_j \cdot \text{size}_j
$$

$$
\text{Flow imbalance} = \frac{V_{\text{buy}} - V_{\text{sell}}}{V_{\text{buy}} + V_{\text{sell}}} \in [-1, 1]
$$

Volume-weighted average prices:

$$
\text{VWAP}_{\text{buy}} = \frac{\sum_{j \in \text{buys}} w_j \cdot \text{size}_j \cdot p_j}{\sum_{j \in \text{buys}} w_j \cdot \text{size}_j}
$$

#### 4.5.4 Signal Generation

A signal fires when flow disagrees with price:

$$
\text{price lean} = p_{\text{current}} - 0.5
$$

$$
\text{flow lean} = \text{flow imbalance}
$$

**Disagreement condition:**

$$
-\text{price lean} \times \text{flow lean} > 0 \quad \text{AND} \quad |\text{flow imbalance}| \geq 0.4
$$

**Edge estimate:**

$$
\text{edge}_{\text{flow}} = |\text{imbalance}| \times |\text{price lean}| \times 0.4
$$

$$
\text{edge}_{\text{VWAP}} = \begin{cases}
\max(p_{\text{current}} - \text{VWAP}_{\text{buy}}, 0) \times 0.3 & \text{if flow lean} > 0 \\
\max(\text{VWAP}_{\text{sell}} - p_{\text{current}}, 0) \times 0.3 & \text{if flow lean} < 0
\end{cases}
$$

$$
\text{edge}_{\text{total}} = \text{edge}_{\text{flow}} + \text{edge}_{\text{VWAP}}
$$

Minimum edge threshold: 2%.

**Parameters:**

| Parameter | Value |
|---|---|
| `large_order_threshold_usd` | $1,000 |
| `flow_window_s` | 3,600s (1 hour) |
| `max_orders_tracked` | 200 per market |
| `min_large_orders` | 3 |
| `flow_disagreement_threshold` | 0.40 |
| `price_disagreement_threshold` | 0.15 |
| `decay_half_life_s` | 1,800s (30 min) |
| `min_edge` | 0.02 |

**Assumptions and Limitations:**
- The $1,000 threshold is arbitrary; optimal threshold depends on market liquidity distribution.
- Not all large orders are informed -- some are hedging or market-making rebalances.
- VWAP-based edge assumes smart money transacts at prices reflecting their information advantage.

**Academic context:** Order flow toxicity and informed trading detection draw from Kyle (1985), VPIN (Easley et al., 2012), and Bouchaud et al. (2018) on order flow and price impact.

---

## 5. Trading Strategies

### 5.1 Implied Probability Arbitrage

**Source:** `strategies/implied_prob_arb.py`

#### 5.1.1 Concept

The simplest and most reliable strategy. In a binary market, the YES and NO prices should sum to approximately 1.0 (modulo the vig/spread). When they deviate:

$$
\text{vig} = (p_{\text{YES}} + p_{\text{NO}}) - 1
$$

- **vig > 0:** Both sides are overpriced (market is extracting excess vig)
- **vig < 0:** Both sides are underpriced (structural discount exists)

#### 5.1.2 Fair Value Computation

$$
p_{\text{fair, YES}} = \frac{p_{\text{YES}}}{p_{\text{YES}} + p_{\text{NO}}}
$$

$$
p_{\text{fair, NO}} = \frac{p_{\text{NO}}}{p_{\text{YES}} + p_{\text{NO}}}
$$

Per-side edge:

$$
\text{edge}_{\text{YES}} = p_{\text{fair, YES}} - p_{\text{YES}}
$$

$$
\text{edge}_{\text{NO}} = p_{\text{fair, NO}} - p_{\text{NO}}
$$

#### 5.1.3 Entry Conditions

1. $\text{vig} < -\text{min\_edge}$ (total prices below 1 by at least 3%)
2. $\text{vig} \leq \text{max\_vig}$ (don't trade when vig is too high)
3. $\text{liquidity} \geq \$200$
4. $\max(\text{edge}_{\text{YES}}, \text{edge}_{\text{NO}}) \geq 0.03$
5. Buy the side with the larger edge

For complement-based mispricing: if the implied fair value from the complement exceeds mid-price by at least 3%, and mid-price < 0.5, a BUY signal is emitted.

#### 5.1.4 Exit Conditions

Close when $|\text{vig}| \leq 0.01$ (edge collapsed).

#### 5.1.5 Signal Strength

$$
\text{strength} = \min\left(\frac{\text{edge}}{0.10}, 1.0\right)
$$

**Parameters:**

| Parameter | Value |
|---|---|
| `min_edge` | 0.03 (3%) |
| `max_vig` | 0.04 (4%) |
| `min_liquidity` | $200 |
| `exit_edge` | 0.01 |

**Assumptions:** Relies on the market converging to $p_{\text{YES}} + p_{\text{NO}} = 1$. This is guaranteed at settlement but may take time pre-settlement.

**Academic context:** The vig/overround concept is standard in sports betting (Shin, 1993) and prediction market microstructure literature.

---

### 5.2 Mean Reversion

**Source:** `strategies/mean_reversion.py`

#### 5.2.1 Hypothesis

After a forecast update (e.g., new information arrival), prediction market prices overshoot in the direction of the news, then revert toward fair value as the market digests the information. This is the prediction market analogue of post-earnings announcement drift and its partial reversal.

#### 5.2.2 Z-Score Computation

Define the spread as the difference between market price and model probability:

$$
s_t = p_{\text{market}, t} - p_{\text{model}, t}
$$

Over a rolling lookback window of $L = 50$ observations:

$$
z_t = \frac{s_t - \bar{s}}{\sigma_s}
$$

#### 5.2.3 Entry Conditions

All of the following must hold:
1. Within post-update window: $0 < t_{\text{elapsed}} \leq 30$ minutes since last update
2. Forecast shift magnitude: $|\Delta f| > 0.5$
3. Z-score extreme: $|z_t| \geq 2.0$
4. Historical reversion rate: $\geq 60\%$ (after 5+ observations)
5. No second update occurred during hold period

**Direction:**
- $z_t > 2.0$: Market overshooting high; SELL (fade the overshoot)
- $z_t < -2.0$: Market overshooting low; BUY

#### 5.2.4 Exit Conditions

- **Z-score reverted:** $|z_t| \leq 0.5$ (success; records reversion)
- **Max holding period:** 20 periods exceeded (timeout; records failure)

#### 5.2.5 Signal Strength

$$
\text{strength} = \min\left(\frac{|z_t|}{4.0}, 1.0\right)
$$

**Parameters:**

| Parameter | Value |
|---|---|
| `zscore_entry` | 2.0 |
| `zscore_exit` | 0.5 |
| `post_update_window_min` | 30 minutes |
| `lookback` | 50 observations |
| `min_reversion_rate` | 60% |
| `max_holding_periods` | 20 |

**Assumptions and Limitations:**
- Requires a model probability (external fair value estimate) to compute the spread.
- The 30-minute window is calibrated for specific information events; different event types may have different optimal windows.
- Reversion rate tracking is per-market, requiring sufficient per-market history.

**Academic context:** Mean reversion in financial markets is extensively documented (Poterba and Summers, 1988; De Bondt and Thaler, 1985). Short-horizon reversal following information shocks is documented in Tetlock (2007) for prediction markets.

---

### 5.3 Volume-Price Divergence

**Source:** `strategies/volume_divergence.py`

#### 5.3.1 Concept

When volume spikes anomalously high but price barely moves, informed participants are likely accumulating a position without moving the market. This is the prediction market analogue of "stealth trading" (Barclay and Warner, 1993).

#### 5.3.2 Volume Z-Score

$$
z_{\text{vol}} = \frac{V_t - \bar{V}}{\sigma_V}
$$

Computed over a rolling lookback of $L = 50$ observations.

#### 5.3.3 Price Drift

$$
\Delta p = p_t - p_{t-5}
$$

(5-period price change)

#### 5.3.4 Divergence Condition

A divergence exists when:
1. $z_{\text{vol}} \geq 2.5$ (volume spike)
2. $|\Delta p| \leq 0.03$ (price hasn't moved proportionally)

#### 5.3.5 Direction and Edge

- If $\Delta p > 0.001$: BUY (follow the small upward drift)
- If $\Delta p < -0.001$: SELL (follow the small downward drift)
- If $|\Delta p| \leq 0.001$: no signal (no directional information)

Edge estimate:

$$
\text{edge} = z_{\text{vol}} \times 0.01
$$

This is a heuristic: higher volume Z-scores imply more informed activity.

#### 5.3.6 Exit Conditions

Close position when volume normalizes: $z_{\text{vol}} < 1.0$.

#### 5.3.7 Signal Strength

$$
\text{strength} = \min\left(\frac{z_{\text{vol}}}{5.0}, 1.0\right)
$$

**Parameters:**

| Parameter | Value |
|---|---|
| `volume_zscore_threshold` | 2.5 |
| `price_move_threshold` | 0.03 (3%) |
| `lookback` | 50 |
| `min_volume` | $100 |

**Assumptions and Limitations:**
- The 0.01x heuristic edge scaling is not calibrated from data; should be validated empirically.
- High volume without price movement could also indicate noise trading, not informed flow.
- Works best in liquid markets where informed traders can accumulate without moving price.

**Academic context:** Volume-price divergence as an information signal follows from the sequential trade models of Easley and O'Hara (1987) and the stealth trading hypothesis of Barclay and Warner (1993).

---

### 5.4 Stale Market Detection

**Source:** `strategies/stale_market.py`

#### 5.4.1 Concept

As a prediction market approaches resolution, its price should converge toward 0 or 1. Markets where the price remains stuck in the middle (far from either boundary) despite approaching expiry are likely mispriced -- the market participants have lost attention, or the event outcome is already largely determined.

#### 5.4.2 Staleness Criteria

A market is classified as "stale" if all three conditions hold:

1. **Low price range:** $\max(p) - \min(p) < 0.05$ over the lookback window
2. **Low price volatility:** $\sigma_p < 0.02$
3. **Near resolution:** TTL $\leq 168$ hours (7 days)

#### 5.4.3 Boundary Distance Filter

Markets already near a boundary ($\min(p, 1-p) < 0.15$) are excluded -- convergence is already underway.

#### 5.4.4 Urgency Factor

$$
\text{urgency} = \max\left(0, 1 - \frac{\text{TTL}}{\text{max\_TTL}}\right)
$$

Urgency increases as expiry approaches, boosting the signal edge.

#### 5.4.5 Direction and Edge

$$
\text{direction} = \begin{cases}
\text{BUY} & \text{if } p > 0.5 \text{ (likely YES)} \\
\text{SELL} & \text{if } p \leq 0.5 \text{ (likely NO)}
\end{cases}
$$

$$
\text{edge} = |p - 0.5| \times \text{urgency} \times 0.3
$$

Minimum edge: 2%.

#### 5.4.6 Exit Conditions

Close position when the market wakes up: price range > 0.10 or price std > 0.04.

**Parameters:**

| Parameter | Value |
|---|---|
| `stale_hours` | 48 |
| `max_ttl_hours` | 168 (7 days) |
| `min_distance_from_boundary` | 0.15 |
| `convergence_threshold` | 0.85 |
| `min_lookback` | 20 observations |

**Assumptions and Limitations:**
- Assumes that price > 0.5 implies likely YES resolution; this is correct on average but not for individual markets.
- Low-activity markets may have stale prices due to low liquidity, not due to mispricing.
- Edge estimate is conservative (0.3x multiplier) to account for uncertainty.

**Academic context:** Market staleness and expiry convergence are documented in Rothschild (2009) and Page and Clemen (2013) for prediction markets.

---

### 5.5 Line Movement / Momentum

**Source:** `strategies/line_movement.py`

#### 5.5.1 Concept

When odds are trending in a direction, the trend may continue as new information is gradually incorporated. This is the prediction market analogue of momentum in equities (Jegadeesh and Titman, 1993) and the "price discovery" process in derivatives.

#### 5.5.2 Linear Regression Model

Fit OLS to the most recent $n$ price observations ($n \geq 20$):

$$
\hat{p}_t = \hat{\beta}_0 + \hat{\beta}_1 t
$$

$$
\hat{\beta}_1 = \frac{\sum (t_i - \bar{t})(p_i - \bar{p})}{\sum (t_i - \bar{t})^2}
$$

$$
R^2 = \frac{\left[\sum (t_i - \bar{t})(p_i - \bar{p})\right]^2}{\sum (t_i - \bar{t})^2 \cdot \sum (p_i - \bar{p})^2}
$$

#### 5.5.3 Entry Conditions

1. $R^2 \geq 0.30$ (trend explains at least 30% of variance)
2. $|\hat{\beta}_1 \times n| \geq 0.03$ (total move over window exceeds 3%)
3. Projected edge: $|\hat{p}_{t+5} - p_t| \geq 0.03$

Where the 5-step projection is:

$$
\hat{p}_{t+5} = \text{clip}(\hat{\beta}_0 + \hat{\beta}_1 \cdot (n + 5), 0.01, 0.99)
$$

Direction follows the slope: BUY if $\hat{\beta}_1 > 0$, SELL if $\hat{\beta}_1 < 0$.

#### 5.5.4 Exit Conditions

Exit when the trend reverses: $|p_t - p_{t-2}| > 0.02$ in the opposite direction of $\hat{\beta}_1$.

#### 5.5.5 Signal Strength

$$
\text{strength} = \min\left(\frac{R^2 \times |\hat{\beta}_1 \times n|}{0.05}, 1.0\right)
$$

**Parameters:**

| Parameter | Value |
|---|---|
| `lookback` | 100 (buffer) |
| `trend_threshold` | 0.03 |
| `momentum_min_r2` | 0.30 |
| `entry_edge` | 0.03 |
| `exit_reversal` | 0.02 |
| `min_data_points` | 20 |

**Assumptions and Limitations:**
- Linear model cannot capture nonlinear momentum (acceleration/deceleration).
- Prediction market trends may be event-driven rather than persistent, leading to false momentum signals.
- The 5-step projection assumes trend continuation, which is inherently uncertain.

**Academic context:** Momentum in prediction markets has been documented by Rothschild and Sethi (2015), though it tends to be weaker than in equity markets due to the binary payoff structure.

---

### 5.6 Cross-Market Consistency Arbitrage

**Source:** `strategies/cross_market_arb.py`

#### 5.6.1 Concept

Logically related markets must satisfy probability constraints. For example:

- **Subset relationship:** $P(\text{Trump wins}) \leq P(\text{Republican wins})$, because Trump winning implies the Republican winning.
- **Temporal relationship:** $P(\text{event by June}) \leq P(\text{event by December})$, because a later deadline is strictly easier to meet.

When these constraints are violated, a directional trade exists.

#### 5.6.2 Contradiction Detection

For a subset relationship between market A (subset) and market B (superset):

$$
\text{gap} = p_A - p_B
$$

If $\text{gap} > 0$ and market A should be $\leq$ market B, a contradiction exists.

For temporal relationships: the market with the earlier deadline should have a lower or equal probability.

#### 5.6.3 Entry Conditions

1. $\text{gap} \geq 0.05$ (5% minimum contradiction)
2. $\text{relationship confidence} \geq 0.50$

Direction:
- Subset violations: SELL the overpriced subset
- Temporal violations: BUY the underpriced later-deadline market

#### 5.6.4 Exit Conditions

Close when $\text{gap} \leq 0.02$.

#### 5.6.5 Signal Strength

$$
\text{strength} = \min\left(\frac{\text{gap}}{0.15}, 1.0\right)
$$

**Parameters:**

| Parameter | Value |
|---|---|
| `min_gap` | 0.05 (5%) |
| `min_confidence` | 0.50 |
| `exit_gap` | 0.02 |

**Assumptions and Limitations:**
- Requires an external relationship graph that correctly identifies logical links between markets.
- Confidence in the relationship is critical; false links produce false signals.
- Single-leg execution means this is a directional bet, not a true arbitrage (one leg may not be tradeable).

**Academic context:** Cross-market arbitrage in prediction markets is a form of statistical arbitrage. The logical constraint framework follows from Bayesian networks (Pearl, 2000) applied to event-linked securities.

---

### 5.7 Orderbook Imbalance

**Source:** `strategies/orderbook_imbalance.py`

#### 5.7.1 Concept

When bid volume significantly exceeds ask volume (or vice versa), short-term directional pressure exists. This signal decays rapidly -- imbalance is most predictive in the first few minutes.

#### 5.7.2 Imbalance Ratio

Aggregate top $D = 5$ price levels of the orderbook:

$$
V_{\text{bid}} = \sum_{l=1}^{D} p_l^{\text{bid}} \times q_l^{\text{bid}}, \quad V_{\text{ask}} = \sum_{l=1}^{D} p_l^{\text{ask}} \times q_l^{\text{ask}}
$$

$$
R = \frac{V_{\text{bid}}}{V_{\text{ask}}}
$$

#### 5.7.3 Time-Decayed Weighted Ratio

Over a history of imbalance snapshots, compute the exponential-decay-weighted average:

$$
\bar{R}_{\text{decay}} = \frac{\sum_{j} w_j R_j}{\sum_{j} w_j}, \quad w_j = \exp\left(-\frac{\ln 2}{\tau_{1/2}} \cdot (t_{\text{now}} - t_j)\right)
$$

With half-life $\tau_{1/2} = 180$s (3 minutes).

#### 5.7.4 Entry Conditions

**BUY signal (bid pressure):**

$$
\bar{R}_{\text{decay}} \geq 3.0
$$

**SELL signal (ask pressure):**

$$
\bar{R}_{\text{decay}} \leq \frac{1}{3.0} \approx 0.333
$$

Minimum depth: $V_{\text{bid}} + V_{\text{ask}} \geq \$100$.

#### 5.7.5 Edge Estimation

$$
\text{edge} = 0.02 \times \ln(\bar{R}_{\text{decay}}) \times \min(2 \times \text{room}, 1.0)
$$

Where "room" is $1 - p$ for upward moves and $p$ for downward moves. The logarithmic scaling reflects diminishing marginal impact of extreme imbalances.

#### 5.7.6 Signal Strength

$$
\text{strength} = \text{clip}\left(\frac{\bar{R}_{\text{decay}} - 3.0}{8.0 - 3.0}, 0.1, 1.0\right)
$$

Saturates at $\bar{R}_{\text{decay}} = 8.0$.

**Parameters:**

| Parameter | Value |
|---|---|
| `imbalance_threshold` | 3.0 (3:1 bid:ask ratio) |
| `decay_half_life_s` | 180s (3 min) |
| `min_depth_usd` | $100 |
| `lookback_window` | 50 snapshots |
| `min_snapshots` | 3 |
| `strength_cap_ratio` | 8.0 |
| `min_edge` | 0.02 |
| `depth_levels` | 5 |

**Assumptions and Limitations:**
- Orderbook data may be stale due to API polling frequency.
- Spoofing (placing large orders that are immediately canceled) can generate false imbalance signals.
- The 3-minute half-life assumes rapid information incorporation; slower markets may need a longer decay.

**Academic context:** Orderbook imbalance as a short-term price predictor is well-documented in traditional markets (Cont et al., 2014; Cartea et al., 2015).

---

### 5.8 Temporal Bias

**Source:** `strategies/temporal_bias.py`

#### 5.8.1 Concept

Three persistent structural biases related to time:

1. **Expiry convergence:** Near-expiry markets should converge to 0 or 1; those stuck in the middle are mispriced.
2. **Time-of-day spreads:** Asian session (00:00-08:00 UTC) has 60% wider spreads due to low US participation.
3. **Weekend drift reversion:** Price drift over weekends reverts on Monday as US participants return.

#### 5.8.2 Sub-Signal 1: Expiry Convergence

Urgency function (exponential as expiry nears):

$$
u = e^{-0.15 \cdot \text{TTL}_{\text{hrs}}}
$$

At TTL = 1 hour: $u \approx 0.86$. At TTL = 24 hours: $u \approx 0.027$.

Edge estimate:

$$
\text{edge} = \left(u \times d \times 0.15 + s_{\text{expected}} \times 0.5\right) \times 1.5
$$

Where $d = \min(p, 1-p)$ is distance from the nearest boundary, $s_{\text{expected}}$ is the session-adjusted spread, and 1.5 is the expiry edge boost.

Fires only when $d \geq 0.10$ (market not yet converging) and TTL $\leq 24$ hours.

Direction: BUY if $p \geq 0.5$, SELL if $p < 0.5$.

#### 5.8.3 Sub-Signal 2: Session Spread Capture

Session spread multipliers (relative to US = 1.0):

| Session | UTC Hours | Multiplier |
|---|---|---|
| Asian | 00:00 - 08:00 | 1.60 |
| European | 08:00 - 13:00 | 1.20 |
| US | 13:00 - 21:00 | 1.00 (baseline) |
| Off-hours | 21:00 - 00:00 | 1.35 |
| Weekend | Any | Additional 1.40x |

The strategy estimates the US-session expected spread by dividing the observed spread by the session multiplier, then captures the excess:

$$
\text{edge} = \frac{s_{\text{observed}} - s_{\text{US expected}}}{2}
$$

Fires when spread ratio $\geq 1.5$ and spread $\geq 3\%$.

#### 5.8.4 Sub-Signal 3: Weekend Drift Mean-Reversion

On Monday before 13:00 UTC (before US session opens):

$$
\text{drift} = p_{\text{Monday}} - p_{\text{Friday close}}
$$

If $|\text{drift}| \geq 0.03$:

$$
\text{edge} = |\text{drift}| \times 0.6
$$

Direction: opposite to drift (mean-reversion). The 0.6 factor reflects the empirical expectation that about 60% of weekend drift reverts.

**Parameters:**

| Parameter | Value |
|---|---|
| `min_edge` | 0.02 |
| `expiry_hours_threshold` | 24 hours |
| `expiry_edge_boost` | 1.5 |
| `spread_ratio_threshold` | 1.5 |
| `weekend_drift_threshold` | 0.03 |
| `min_liquidity` | $200 |
| `off_session_min_spread` | 0.03 |

**Assumptions and Limitations:**
- Session effects assume a predominantly US-based participant base; as Polymarket grows globally, these patterns may weaken.
- Weekend drift reversion requires a Friday close price, which may not always be available.
- Expiry convergence assumes the "truth" is close to current price direction, which is reasonable for large markets but may fail for illiquid or manipulated ones.

**Academic context:** Time-of-day effects in financial markets are well-documented (Harris, 1986; Admati and Pfleiderer, 1988). Weekend effects and Monday reversals are studied in Keim and Stambaugh (1984). Expiry convergence in prediction markets follows from the law of iterated expectations.

---

## 6. Portfolio Construction

### 6.1 Kelly Criterion Sizing

For each trade, position size is determined by a fractional Kelly criterion:

$$
f^* = \frac{p \cdot b - (1 - p)}{b}
$$

Where:
- $p = \min(p_{\text{fair}} + \text{edge}, 0.99)$ is the estimated true probability
- $b = \frac{1}{p_{\text{market}}} - 1$ are the implied odds
- The system uses quarter-Kelly: $f = 0.25 \times f^*$

Quarter-Kelly is chosen over full or half-Kelly to reduce variance. Expected growth rate at quarter-Kelly is approximately 75% of full Kelly's expected growth with dramatically reduced drawdowns (Thorp, 2006).

### 6.2 Position Size Constraints

$$
\text{size} = \text{clip}(f \times \text{cash}, \$0.50, \min(\text{cash} \times 0.10, \$5.00))
$$

Multiple hard limits apply simultaneously:

| Constraint | Value |
|---|---|
| Kelly fraction | 0.25 (quarter-Kelly) |
| Minimum trade size | $0.50 |
| Maximum trade size | $5.00 |
| Maximum per-position | $5.00 |
| Maximum open positions | 15 |
| Maximum trades per scan | 5 |
| Maximum single-position as % of cash | 10% |
| Maximum daily drawdown | 20% |
| Starting capital | $50.00 |

### 6.3 Entry Filters

Before any position is opened:
1. $\text{edge} \geq 3.5\%$ (after calibration)
2. $\text{confidence} \geq 0.30$ (from Markov model)
3. $8\% \leq p \leq 92\%$ (avoid extreme prices)
4. $\text{spread} \leq 10\%$
5. Recent volatility $\sigma_{20} \leq 25\%$ (avoid chaotic markets)
6. Minimum 15 price history points
7. Kelly fraction $f > 0$

### 6.4 Edge Asymmetry

The system applies asymmetric edge multipliers favoring the NO side, consistent with the empirical NO-side premium:

- **SELL (NO-side):** total edge = $|e_{\text{calibrated}}| \times 1.2 + e_{\text{NO}} \times 0.4$
- **BUY (YES-side):** total edge = $e_{\text{calibrated}} \times 1.2 \times 0.7$

The 0.7 discount on YES reflects the systematic YES overpricing from retail flow.

### 6.5 Fee Accounting

All PnL calculations deduct a 2% taker fee:

$$
\text{PnL}_{\text{net}} = \text{PnL}_{\text{gross}} - \text{size} \times 0.02
$$

---

## 7. Backtesting Methodology

### 7.1 Walk-Forward Framework

The system uses walk-forward optimization to prevent overfitting:

1. **Training window:** Fit parameters on historical data
2. **Validation window:** Test on unseen subsequent data
3. **Roll forward:** Shift both windows and repeat

This ensures all reported performance metrics are truly out-of-sample.

### 7.2 Bootstrap Confidence Intervals

Performance metrics (win rate, Sharpe ratio, cumulative return) are accompanied by bootstrap confidence intervals:

$$
\hat{\theta}^{(b)} = \text{statistic}(\text{resample}(\text{trades}, n, \text{replace}=\text{True})), \quad b = 1, \ldots, B
$$

95% CI: $[\hat{\theta}^{(0.025)}, \hat{\theta}^{(0.975)}]$

### 7.3 Anti-Lookahead Safeguards

The simulation environment (Section 8) enforces strict temporal isolation:
- No access to future price data
- No access to resolution outcomes during evaluation
- Fills at actual book prices after real computation latency
- Markets cannot be re-entered after closing (prevents memorization)
- Markov model cache is reset between runs

### 7.4 Reported Metrics

| Metric | Headline Value |
|---|---|
| Win rate | 82.4% |
| Cumulative return | +301% |
| Sharpe ratio | 3.95 |
| Universe | 200 markets |
| Variant | v2_no_maker |

---

## 8. Simulated Environment Architecture

**Source:** `backtesting/sim_environment.py`

### 8.1 Design Principles

The simulation environment enforces **true isolation** between the market data source and the trading bot. This is not a vectorized backtest -- it is a simulated live environment with real computation latency.

### 8.2 Three-Component Architecture

**Component 1: Market Simulator.** Reads PMXT parquet files containing full orderbook events, advances a simulation clock, and exposes data only up to the current timestamp via a mock API that mirrors the real Polymarket CLOB API.

**Component 2: Trading Bot.** Has NO access to the simulator's internal state, the parquet file, or any future events. It calls `sim.get_markets()`, `sim.get_book()`, `sim.get_price_history()` -- identical signatures to the live API.

**Component 3: Orchestrator.** Connects the simulator and bot, advancing the simulation clock and triggering bot scan cycles.

### 8.3 Latency and Slippage Realism

Key realism features:
1. **Real computation latency:** When the bot's Markov model takes 250ms to evaluate, 250ms of simulated market events pass.
2. **Fill at current book:** Orders fill at the best ask/bid at fill time, not at evaluation time. The book may have moved during computation.
3. **Slippage measurement:** $\text{slippage (bps)} = |p_{\text{fill}} - p_{\text{signal}}| \times 10{,}000$

### 8.4 Playback Modes

| Mode | `speed` | Behavior |
|---|---|---|
| Fast | 0 | Process events as fast as possible; real latency still applies |
| Realtime | 1 | 1 hour of data = 1 hour of wall clock |
| Accelerated | 60 | 1 hour of data = 1 minute of wall clock |

### 8.5 Anti-Memorization Safeguards

- `self._closed_markets`: Once a market is exited, it is never re-entered. This prevents the bot from "learning" which markets win.
- `self.markov.reset()`: Cache is cleared at bot initialization.
- Price history is sampled (every 10th update) to prevent information leakage from high-frequency data.
- Maximum 50 trades per run prevents exhaustive scanning.

### 8.6 Exit Mechanics

Positions exit when market price converges to extremes ($p \geq 0.97$ or $p \leq 0.03$), simulating resolution:

$$
\text{PnL} = \text{shares} \times p_{\text{resolution}} - \text{cost} - \text{fee}
$$

Where fee = 2% of size.

---

## 9. Known Risks and Overfitting Assessment

### 9.1 Overfitting Concerns

| Risk | Mitigation |
|---|---|
| Strategy parameters tuned to historical data | Walk-forward validation; quarter-Kelly sizing limits damage from poor periods |
| Multiple comparisons (8 strategies tested) | Bonferroni-style skepticism; effective-N < raw count due to correlation |
| Data-mined calibration table (72.1M trades) | Calibration is derived from a different study, not this system's backtest |
| Markov model sees historical prices | Model uses only prices up to "current" time in simulation |
| Simulation doesn't model market impact | Position sizes are small ($0.50-$5.00); impact at this scale is negligible |

### 9.2 Regime Risk

- **Platform maturation:** As Polymarket attracts more sophisticated participants, structural inefficiencies (longshot bias, session effects) will compress. Category multipliers and NO-side premium may diminish.
- **Fee changes:** The 2% taker fee is hardcoded. Fee increases would erode edge on low-profit-per-trade strategies.
- **Regulatory risk:** Prediction market legality varies by jurisdiction. Regulatory action could impact liquidity or platform availability.

### 9.3 Execution Risk

- **Partial fills:** Combinatorial arbitrage requires all legs to execute. Partial fills create unhedged directional exposure.
- **API latency:** Real CLOB API latency may exceed simulation estimates.
- **Orderbook spoofing:** Imbalance strategy is vulnerable to large orders placed then immediately canceled.

### 9.4 Model Risk

- **Markov property assumption:** Real prediction market prices are not memoryless. Auto-correlation, momentum, and clustering violate the Markov assumption.
- **Beta distribution independence:** The Bayesian updater treats trades as i.i.d. Bernoulli trials, ignoring serial correlation in outcomes.
- **Linear momentum model:** The line movement strategy fits a linear trend to what may be nonlinear dynamics.
- **Semantic clustering errors:** TF-IDF-based market clustering (Tier 3) may produce false links, leading to combinatorial arbitrage signals on unrelated markets.

### 9.5 Survivorship Bias

Backtesting on PMXT data includes only markets that generated orderbook activity. Markets that were listed but never traded are excluded, potentially biasing results upward.

---

## 10. Parameter Registry

Every tunable parameter in the system, with current production values.

### 10.1 Global Entry/Exit

| Parameter | Value | Source |
|---|---|---|
| `ENTRY_THRESHOLD` | 0.15 | `config/settings.py` |
| `EXIT_THRESHOLD` | 0.45 | `config/settings.py` |
| `MIN_EDGE` | 0.03 | `config/settings.py` |
| `MIN_CONFIDENCE` | 0.70 | `config/settings.py` |
| `STOP_LOSS_RATIO` | 0.50 | `config/settings.py` |

### 10.2 Position Sizing

| Parameter | Value | Source |
|---|---|---|
| `KELLY_FRACTION` | 0.25 | `config/settings.py` |
| `MIN_TRADE_SIZE` | $0.50 | `config/settings.py` |
| `MAX_TRADE_SIZE` | $5.00 | `config/settings.py` |
| `MAX_POSITION_USD` | $5.00 | `config/settings.py` |
| `MAX_OPEN_POSITIONS` | 15 | `config/settings.py` |
| `MAX_TRADES_PER_RUN` | 5 | `config/settings.py` |
| `MAX_DAILY_DRAWDOWN` | 0.20 | `config/settings.py` |
| `STARTING_CAPITAL` | $50.00 | `config/settings.py` |

### 10.3 Market Filters

| Parameter | Value | Source |
|---|---|---|
| `MIN_MARKET_LIQUIDITY` | $100 | `config/settings.py` |
| `MIN_MARKET_VOLUME_24H` | $50 | `config/settings.py` |
| `MAX_MARKET_SPREAD` | 0.15 | `config/settings.py` |
| `CLOB_TOP_N_TRACKED` | 50 | `config/settings.py` |
| `MAX_TTL_DAYS` | 90 | `config/settings.py` |
| `MIN_TTL_HOURS` | 2 | `config/settings.py` |

### 10.4 Scan Timing

| Parameter | Value | Source |
|---|---|---|
| `SCAN_INTERVAL` | 120s | `config/settings.py` |
| `GAMMA_POLL_INTERVAL` | 300s | `config/settings.py` |
| `CLOB_POLL_INTERVAL` | 20s | `config/settings.py` |
| `FULL_SCAN_INTERVAL` | 3600s | `config/settings.py` |

### 10.5 Strategy-Specific Parameters

#### Implied Probability Arbitrage
| Parameter | Value |
|---|---|
| `min_edge` | 0.03 |
| `max_vig` | 0.04 |
| `min_liquidity` | $200 |
| `exit_edge` | 0.01 |

#### Mean Reversion
| Parameter | Value |
|---|---|
| `zscore_entry` | 2.0 |
| `zscore_exit` | 0.5 |
| `post_update_window_min` | 30 |
| `lookback` | 50 |
| `min_reversion_rate` | 0.60 |
| `max_holding_periods` | 20 |

#### Volume Divergence
| Parameter | Value |
|---|---|
| `volume_zscore_threshold` | 2.5 |
| `price_move_threshold` | 0.03 |
| `lookback` | 50 |
| `min_volume` | $100 |

#### Stale Market
| Parameter | Value |
|---|---|
| `stale_hours` | 48 |
| `max_ttl_hours` | 168 |
| `min_distance_from_boundary` | 0.15 |
| `convergence_threshold` | 0.85 |
| `min_lookback` | 20 |

#### Line Movement
| Parameter | Value |
|---|---|
| `lookback` | 100 |
| `trend_threshold` | 0.03 |
| `momentum_min_r2` | 0.30 |
| `entry_edge` | 0.03 |
| `exit_reversal` | 0.02 |
| `min_data_points` | 20 |

#### Cross-Market Arbitrage
| Parameter | Value |
|---|---|
| `min_gap` | 0.05 |
| `min_confidence` | 0.50 |
| `exit_gap` | 0.02 |

#### Orderbook Imbalance
| Parameter | Value |
|---|---|
| `imbalance_threshold` | 3.0 |
| `decay_half_life_s` | 180 |
| `min_depth_usd` | $100 |
| `lookback_window` | 50 |
| `min_snapshots` | 3 |
| `strength_cap_ratio` | 8.0 |
| `min_edge` | 0.02 |
| `depth_levels` | 5 |

#### Temporal Bias
| Parameter | Value |
|---|---|
| `min_edge` | 0.02 |
| `expiry_hours_threshold` | 24.0 |
| `expiry_edge_boost` | 1.5 |
| `spread_ratio_threshold` | 1.5 |
| `weekend_drift_threshold` | 0.03 |
| `min_liquidity` | $200 |
| `off_session_min_spread` | 0.03 |

### 10.6 Core Model Parameters

#### Markov Model
| Parameter | Value |
|---|---|
| `n_states` | 10 |
| `n_simulations` | 10,000 (2,000 in sim) |
| `default_horizon_days` | 30 |

#### Alpha Combiner
| Parameter | Value |
|---|---|
| `min_trades_for_ic` | 20 |
| `ic_window` | 100 |
| `regularization` | 0.10 |
| `min_ic_threshold` | -0.05 |

#### Bayesian Updater
| Parameter | Value |
|---|---|
| `prior_alpha` | 1.0 |
| `prior_beta` | 1.0 |
| `ic_window` | 100 |
| `min_observations` | 10 |
| `decay_factor` | 0.995 |
| `regime_sensitivity` | 2.0 |

#### Combinatorial Arbitrage
| Parameter | Value |
|---|---|
| `min_profit_pct` | 0.5% |
| `min_link_confidence` | 0.60 |
| `max_legs` | 6 |
| `fee_buffer` | 1.20 |

#### Smart Money Detector
| Parameter | Value |
|---|---|
| `large_order_threshold_usd` | $1,000 |
| `flow_window_s` | 3,600 |
| `max_orders_tracked` | 200 |
| `min_large_orders` | 3 |
| `flow_disagreement_threshold` | 0.40 |
| `decay_half_life_s` | 1,800 |
| `min_edge` | 0.02 |

### 10.7 Execution Parameters

| Parameter | Value | Source |
|---|---|---|
| `ORDER_TIMEOUT` | 30s | `config/settings.py` |
| `ORDER_POLL_INTERVAL` | 1.0s | `config/settings.py` |
| `ORDER_POLL_MAX_INTERVAL` | 5.0s | `config/settings.py` |
| `ORDER_POLL_BACKOFF` | 1.5x | `config/settings.py` |
| `ORDER_MAX_POLL_ERRORS` | 5 | `config/settings.py` |
| `SIMULATED_SLIPPAGE` | 0.005 | `config/settings.py` |

### 10.8 Bias Calibration Constants

| Parameter | Value |
|---|---|
| `MAKER_EDGE_PER_TRADE` | +1.12% |
| `TAKER_COST_PER_TRADE` | -1.12% |
| `NO_SIDE_PREMIUM` | 1.5% average |
| `TAKER_FEE_RATE` | 2% (standard) |
| `CRYPTO_FEE_RATE` | 5% (crypto markets) |

---

## 11. References

1. **Admati, A. R. and Pfleiderer, P.** (1988). A theory of intraday patterns: Volume and price variability. *Review of Financial Studies*, 1(1), 3-40.

2. **Ang, A. and Bekaert, G.** (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

3. **Barclay, M. J. and Warner, J. B.** (1993). Stealth trading and volatility: Which trades move prices? *Journal of Financial Economics*, 34(3), 281-305.

4. **Bouchaud, J.-P., Bonart, J., Donier, J., and Gould, M.** (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope*. Cambridge University Press.

5. **Cartea, A., Jaimungal, S., and Penalva, J.** (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

6. **Chapelle, O. and Li, L.** (2011). An empirical evaluation of Thompson Sampling. *Advances in Neural Information Processing Systems*, 24.

7. **Cont, R., Kukanov, A., and Stoikov, S.** (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.

8. **De Bondt, W. F. M. and Thaler, R.** (1985). Does the stock market overreact? *Journal of Finance*, 40(3), 793-805.

9. **Easley, D., Lopez de Prado, M. M., and O'Hara, M.** (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457-1493.

10. **Easley, D. and O'Hara, M.** (1987). Price, trade size, and information in securities markets. *Journal of Financial Economics*, 19(1), 69-90.

11. **Griffith, R. M.** (1949). Odds adjustments by American horse-race bettors. *American Journal of Psychology*, 62(2), 290-294.

12. **Grinold, R. C. and Kahn, R. N.** (2000). *Active Portfolio Management*. McGraw-Hill, 2nd edition.

13. **Hamilton, J. D.** (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

14. **Harris, L.** (1986). A transaction data study of weekly and intradaily patterns in stock returns. *Journal of Financial Economics*, 16(1), 99-117.

15. **Jegadeesh, N. and Titman, S.** (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.

16. **Keim, D. B. and Stambaugh, R. F.** (1984). A further investigation of the weekend effect in stock returns. *Journal of Finance*, 39(3), 819-835.

17. **Kyle, A. S.** (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.

18. **Leigh, A., Wolfers, J., and Zitzewitz, E.** (2002). What do financial markets think of war in Iraq? *NBER Working Paper No. 9587*.

19. **Page, L. and Clemen, R. T.** (2013). Do prediction markets produce well-calibrated probability forecasts? *Economic Journal*, 123(568), 491-513.

20. **Pearl, J.** (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

21. **Poterba, J. M. and Summers, L. H.** (1988). Mean reversion in stock prices: Evidence and implications. *Journal of Financial Economics*, 22(1), 27-59.

22. **Rothschild, D.** (2009). Forecasting elections: Comparing prediction markets, polls, and their biases. *Public Opinion Quarterly*, 73(5), 895-916.

23. **Rothschild, D. and Sethi, R.** (2015). Wishful thinking and prediction markets. Working paper, Microsoft Research.

24. **Shin, H. S.** (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *Economic Journal*, 103(420), 1141-1153.

25. **Snowberg, E. and Wolfers, J.** (2010). Explaining the favorite-longshot bias: Is it risk-love or misperceptions? *Journal of Political Economy*, 118(4), 723-746.

26. **Tetlock, P. C.** (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139-1168.

27. **Thaler, R. H. and Ziemba, W. T.** (1988). Anomalies: Parimutuel betting markets: Racetracks and lotteries. *Journal of Economic Perspectives*, 2(2), 161-174.

28. **Thorp, E. O.** (2006). The Kelly criterion in blackjack, sports betting, and the stock market. In *Handbook of Asset and Liability Management*, Vol. 1.

29. **arXiv:2508.03474.** Multi-market arbitrage on Polymarket: $39.6M in extractable value from probability constraint violations. (2025).

---

*Document generated 2026-04-10. All formulas and parameters correspond to the codebase at commit `9d6f972`.*
