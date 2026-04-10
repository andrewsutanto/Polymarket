# Alpha Research: Prediction Market Trading Strategies

> Compiled 2026-04-10. Sources: academic papers, arxiv preprints, industry research.
> Focus: actionable strategies for Polymarket CLOB trading.

---

## Tier 1: Implement Immediately

These strategies have strong empirical backing, proven edge, and are directly feasible on Polymarket.

### 1. Longshot Bias Exploitation (Sell Overpriced Tails)

**Source:** Becker (2025), "The Microstructure of Wealth Transfer in Prediction Markets" -- 72.1M trades on Kalshi.

**Core idea:** Contracts priced below 20c are systematically overpriced (takers overpay for YES longshots). Contracts above 80c are underpriced. Selling NO at extreme prices or buying high-probability YES contracts captures this persistent bias.

**Quantified edge:**
- 1c contracts win 0.43% vs 1% implied = -57% mispricing
- 5c contracts win 4.18% vs 5% implied = -16.4% mispricing
- 95c contracts win 95.83% vs 95% implied = +0.87% mispricing
- Takers lose -1.12% avg; makers gain +1.12% avg

**Category variation (maker-taker gap in pp):**
- Finance: 0.17 (efficient)
- Politics: 1.02
- Sports: 2.23
- Crypto: 2.69
- Entertainment: 4.79
- World Events: 7.32

**Execution on Polymarket:** YES. Post limit orders on the maker side at extreme prices (sub-10c, above 90c). Focus on Sports, Crypto, Entertainment, World Events for maximum edge. Avoid Finance markets.

**Entry rule:** Post limit sell orders for YES tokens at prices <= 10c, or buy YES at prices >= 85c where your model agrees with the favorite.
**Exit rule:** Hold to resolution (binary settlement).

**Data requirements:** Polymarket CLOB API for orderbook depth; historical resolution rates by price bucket.
**Implementation complexity:** Low.
**Tested live:** Yes -- Becker's 72.1M trade dataset confirms. Makers earn +1.12% excess returns systematically.

---

### 2. Passive Market Making (Liquidity Provision)

**Source:** Becker (2025); Whelan (2025), "The Economics of the Kalshi Prediction Market"; multiple industry analyses.

**Core idea:** Makers systematically earn from takers. The edge is NOT directional forecasting -- it is structural. Post two-sided quotes, earn the spread, and let the longshot bias work in your favor. Wealth transfers from takers to makers regardless of direction.

**Quantified edge:** +1.12% excess return per trade on average. Bots achieve 85%+ win rate with $206K+ profits vs human takers losing money.

**Execution on Polymarket:** YES. Use CLOB API to post bid/ask quotes. Polymarket has zero taker fees on standard markets, making maker strategies especially attractive.

**Entry rule:** Post symmetric bid/ask around estimated fair value. Spread = f(volatility, time-to-expiry, category). Minimum 2c spread on illiquid markets, 1c on liquid.
**Exit rule:** Inventory management -- flatten when net delta exceeds threshold. Widen spreads when inventory is skewed.

**Risk management:** Binary settlement means inventory risk is catastrophic if wrong. Use Stoikov-adapted model with binary settlement adjustment. Maximum position per market = 2-5% of bankroll.

**Data requirements:** CLOB API real-time data, historical volatility by market type.
**Implementation complexity:** Medium.
**Tested live:** Yes -- bots dominate Polymarket profits; 27% of bot profits come from non-arbitrage strategies including market making.

---

### 3. Intra-Market Dutch Book Arbitrage

**Source:** IMDEA Networks (2025), "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets" -- $40M+ profits documented Apr 2024-Apr 2025.

**Core idea:** When mutually exclusive outcomes in a single market don't sum to $1.00, you buy the complete set for guaranteed profit. Example: if YES costs 48c and NO costs 49c, buy both for 97c, guaranteed $1.00 payout = 3c risk-free.

**Quantified edge:** $40M+ extracted from Polymarket alone in one year. Median spread: 0.3%. Average duration: 2.7 seconds.

**Execution on Polymarket:** YES, but highly competitive. Need sub-second execution. Most opportunities are captured by existing bots.

**Entry rule:** Monitor all markets for YES + NO < $1.00 (accounting for fees). Execute when spread > fee threshold.
**Exit rule:** Immediate -- buy both sides atomically.

**Data requirements:** Real-time CLOB data, websocket feeds.
**Implementation complexity:** Low (logic simple), but HIGH infrastructure requirements (latency).
**Tested live:** Yes. Proven at scale. However, 62% of combinatorial arb attempts fail to profit due to execution risk.

---

### 4. Political Market Underconfidence (Buy Extremes in Politics)

**Source:** Le (2025), "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets" -- 292M trades, 327K contracts.

**Core idea:** Political markets on Polymarket and Kalshi exhibit persistent underconfidence -- prices are chronically compressed toward 50%. A contract trading at 80c in politics has >80% true probability. Buy high-confidence political outcomes at their market price.

**Quantified edge:** Calibration decomposition explains 87.3% of variance. Political underconfidence is the dominant bias. Markets with >100 days to expiry show 4.7-10.9pp systematic bias.

**Execution on Polymarket:** YES. Identify political markets where your model says true probability is significantly higher/lower than market price. Buy the extreme (high-conviction) side.

**Entry rule:** When market price is >75c and your model says >85c, buy YES. When market price is <25c and model says <15c, buy NO. Minimum 10pp edge required.
**Exit rule:** Hold to resolution, or exit if market moves to within 3pp of your fair value.

**Data requirements:** Poll aggregation model, fundamentals model for political events.
**Implementation complexity:** Low-Medium (need a forecasting model).
**Tested live:** Yes -- calibration bias is documented across both Kalshi and Polymarket.

---

### 5. Kelly Criterion Position Sizing

**Source:** Beygelzimer et al., "Learning Performance of Prediction Markets with Kelly Bettors"; arxiv 2412.14144; Bawa (2025).

**Core idea:** Optimal position sizing for binary prediction markets using the Kelly formula: f* = (bp - q) / b, where b = (1 - market_price) / market_price. Use fractional Kelly (0.25x-0.5x) for robustness.

**Worked example:** Contract at $0.60, your estimate = 75%. b = 0.667, f* = 37.5%. At 0.25x Kelly, bet 9.4% of bankroll.

**Key insight from paper:** Prices in prediction markets do NOT equal probabilities due to payout asymmetries. Maximum gap between price and mean belief can be extreme. Kelly formula must account for this: f = (Q - P)/(1 + Q) where P and Q are odds ratios.

**Execution on Polymarket:** YES. Apply to every trade as position sizing overlay.
**Implementation complexity:** Low.
**Tested live:** Theoretical framework with strong empirical support in betting markets.

---

## Tier 2: Worth Testing

Promising strategies with moderate evidence or complexity. Backtest before deploying capital.

### 6. Cross-Platform Arbitrage (Polymarket vs Kalshi)

**Source:** Multiple papers; IMDEA Networks (2025); industry guides.

**Core idea:** Same event trades at different prices on Polymarket and Kalshi. Buy YES on cheaper platform, NO on the other. Documented 2-4% median deviations on equivalent markets, with 1/3 of pairs showing persistent directional disagreement.

**Quantified edge:** One mechanical strategy achieved 1,218% cumulative return over 800 days entering the highest-yield arb. Typical 1-3% per trade risk-free.

**Feasibility issues:**
- Need accounts on both platforms (Kalshi = US-only, CFTC regulated)
- Non-atomic execution across platforms (75% of orders fill within ~1 hour)
- Capital locked until resolution (can be months)
- Different resolution semantics create basis risk (e.g., Kalshi uses Central Park weather data, Polymarket uses LaGuardia)

**Entry rule:** YES(A) + NO(B) < $1.00 minus fees (need >2.5c gross spread).
**Exit rule:** Hold to resolution.

**Data requirements:** Real-time data from both platforms, semantic matching of equivalent markets.
**Implementation complexity:** High.
**Tested live:** Yes, profitable at scale, but capital-intensive and operationally complex.

---

### 7. Order Book Imbalance (OBI) Signal

**Source:** Cont et al. (2014); Kolm et al. (2023); multiple HFT papers.

**Core idea:** Order book imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume). Strong linear relationship between OBI and short-term price moves. When buy pressure dominates, price tends to rise.

**Polymarket adaptation:** Monitor CLOB depth via API. When OBI > threshold, lean directionally. Use as alpha signal for market making (skew quotes in direction of imbalance).

**Quantified edge:** In equity markets, OBI predicts next-tick direction with 55-65% accuracy. Prediction market CLOB likely less efficient, potentially higher signal.

**Caveats:** Spoofing and iceberg orders can distort OBI. Polymarket CLOB may have thin books making signal noisy.

**Entry rule:** When OBI > +0.3, buy (lean bid). When OBI < -0.3, sell (lean ask).
**Exit rule:** Mean reversion of OBI toward 0, or time-based exit (minutes).

**Data requirements:** Real-time L2 orderbook data from CLOB API.
**Implementation complexity:** Medium.
**Tested live:** Proven in equities/crypto; untested specifically on Polymarket CLOB.

---

### 8. VPIN / Flow Toxicity Detection

**Source:** Easley, Lopez de Prado, O'Hara (2012); Kalshi smart-money detection library.

**Core idea:** VPIN measures the probability of informed trading by tracking volume imbalance between buy and sell-initiated trades. High VPIN = informed traders are present = adverse selection risk for market makers. Low VPIN = safe to provide liquidity.

**Polymarket application:**
- When VPIN is LOW: widen market making activity, capture spread safely.
- When VPIN is HIGH: pull quotes, or follow the informed flow direction.
- Use as a regime filter for Strategy #2 (market making).

**Data requirements:** Trade-by-trade data with aggressor side identification. Polymarket CLOB provides this.
**Implementation complexity:** Medium.
**Tested live:** Open-source implementation exists (Kalshi smart-money detection on GitHub). Predicted Flash Crash 1 hour early.

---

### 9. Whale/Smart Money Following

**Source:** Polymarket on-chain analytics ecosystem; wallet tracking platforms.

**Core idea:** Track wallets of consistently profitable traders. When multiple smart wallets converge on the same position, follow. Only 12.7% of Polymarket users are profitable -- the rest are noise.

**Improved approach (wallet baskets):** Create topic-specific baskets of 5-10 proven wallets. Enter when >80% of basket wallets take the same side within a tight price band.

**Red flags:** Don't copy HFT bots (their edge is speed, not direction). Don't copy BTC/ETH 15-min market traders (spread capture, not replicable). Look for wallets with <100 trades/month, 2%+ returns, high gain/loss ratio.

**Data requirements:** On-chain trade data, wallet P&L history, alert systems (sub-2s latency available from tools like Polyburg, Polywhaler).
**Implementation complexity:** Medium.
**Tested live:** Partially. Community reports mixed results. Wallet baskets outperform single-wallet copying.

---

### 10. Sentiment-Driven Momentum

**Source:** Bollen & Mao (2011); CEPR research; multiple NLP papers.

**Core idea:** Twitter/X sentiment predicts short-term market moves. FinBERT and LLM-based sentiment models on social media posts about prediction market topics can generate alpha signals, especially for politics and entertainment markets where retail sentiment drives prices.

**Polymarket application:** Monitor Twitter/X, Reddit, and news for sudden sentiment shifts on active markets. Enter positions before the market fully incorporates the sentiment information.

**Quantified edge:** Twitter mood predicted DJIA with 87.6% accuracy (Bollen & Mao). Prediction market application is untested but likely higher-alpha due to thinner liquidity and slower price adjustment.

**Entry rule:** Sentiment spike (z-score > 2 from rolling mean) in direction of position. Enter market order.
**Exit rule:** Sentiment mean-reversion (z-score returns to < 0.5) or 24-hour time stop.

**Data requirements:** Twitter/X API or scraper, NLP model (FinBERT), real-time processing pipeline.
**Implementation complexity:** High.
**Tested live:** Proven in equity markets; untested on prediction markets specifically.

---

### 11. Overreaction Mean Reversion

**Source:** Restocchi et al. (2019), "Improving prediction market forecasts by detecting and correcting possible over-reaction to price movements."

**Core idea:** Prediction markets overreact to new information in the short term, then revert. After a large price move (>10pp in <1 hour), the market tends to partially revert within 24 hours. Fade extreme moves.

**Entry rule:** When price moves >10pp in <1 hour, take contrarian position at 50% of move magnitude.
**Exit rule:** Exit when price reverts 40-60% of the move, or 24-hour time stop.

**Data requirements:** Tick-level price data with timestamps.
**Implementation complexity:** Low.
**Tested live:** Documented in academic literature for prediction markets. Magnitude varies by market type.

---

### 12. Model-vs-Market Divergence (Superforecasting Edge)

**Source:** Tetlock (2015); Wharton research; a16z analysis.

**Core idea:** Build a quantitative model (poll aggregation + fundamentals) and trade when model probability diverges significantly from market price. Combined forecasts (polls + markets + models) reduce error by 16-59% vs any single method. When your model disagrees with the market by >10pp, the model is often right.

**Polymarket application:** Build models for specific domains (elections, economic data, sports). When model output diverges from market price by >10pp after accounting for the known biases (political underconfidence, longshot bias), enter a position.

**Entry rule:** |Model_probability - Market_price| > 10pp. Kelly-size the position.
**Exit rule:** Market converges to within 3pp of model, or model updates change the signal.

**Data requirements:** Domain-specific data (polls, economic indicators, sports stats), model infrastructure.
**Implementation complexity:** High (per domain).
**Tested live:** Superforecasters match/beat market accuracy. Combined models outperform both.

---

## Tier 3: Long-Term Research

Interesting concepts requiring significant R&D or infrastructure investment.

### 13. Combinatorial Arbitrage Across Related Markets

**Source:** IMDEA Networks (2025); Bawa (2025) -- analysis of 86M trades.

**Core idea:** Exploit logical dependencies between related markets. Example: "Will X win the primary?" and "Will X win the general?" must satisfy P(general) <= P(primary). When they don't, arbitrage exists.

**Reality check:** 62% of combinatorial arb attempts fail. Captures only 0.24% of total profits despite 10x implementation complexity. Non-atomic execution (fills take ~1 hour) destroys edge. Requires LLM-based semantic analysis to identify relationships.

**Implementation complexity:** Very High.
**Tested live:** Yes, but poor results at scale.

---

### 14. Bayesian Network Pricing for Combinatorial Markets

**Source:** Sontag et al. (2012), "Probability and Asset Updating using Bayesian Networks for Combinatorial Prediction Markets."

**Core idea:** Use Bayesian networks to model dependencies between related prediction markets. Update probabilities in real-time using junction tree inference. Trade markets that are mispriced relative to the network's conditional probability estimates.

**Implementation complexity:** Very High.
**Tested live:** Academic only.

---

### 15. Time Decay / Horizon Effect Trading

**Source:** Le (2025); Interest-bearing position research.

**Core idea:** Markets with >100 days to expiration show 4.7-10.9pp systematic bias. Long-horizon markets compress toward 50% because capital is locked. As expiration approaches, prices should converge to true probability, creating a "theta-like" drift toward extremes.

**Strategy:** Buy high-conviction outcomes early in long-duration markets at the compressed price, earn the convergence as resolution approaches.

**Implementation complexity:** Medium (capital lockup is the main cost).
**Tested live:** Documented bias, but trading it requires patience and capital.

---

### 16. Latency Arbitrage (Crypto-Linked Markets)

**Source:** Industry analysis; bot profitability research.

**Core idea:** Polymarket prices for crypto-linked events (BTC/ETH price targets) lag confirmed spot price movements on Binance/Coinbase. Race to update prediction market prices before the CLOB adjusts.

**Reality check:** Window is milliseconds. Requires co-located infrastructure. Already dominated by professional HFT bots. Likely negative expected value for new entrants.

**Implementation complexity:** Very High.
**Tested live:** Yes, but saturated.

---

### 17. Event-Driven News Trading

**Source:** General event-driven literature; prediction market specific analysis.

**Core idea:** Trade prediction markets immediately after material news breaks (economic data releases, political developments, court rulings). The edge is speed of interpretation, not speed of execution.

**Key insight for Polymarket:** Markets often take 5-30 minutes to fully incorporate breaking news on illiquid markets. If you can interpret the news impact faster than the median trader, there's a window.

**Implementation complexity:** Medium (need news feeds + automated interpretation).
**Tested live:** General concept proven; prediction market specific results mixed.

---

## Summary Matrix

| # | Strategy | Edge (est.) | Complexity | Feasibility | Priority |
|---|----------|-------------|------------|-------------|----------|
| 1 | Longshot Bias | 1-16% per trade | Low | High | **T1** |
| 2 | Market Making | +1.12% avg | Medium | High | **T1** |
| 3 | Dutch Book Arb | 0.3% median | Low/High infra | Medium | **T1** |
| 4 | Political Underconfidence | 5-10pp | Low-Med | High | **T1** |
| 5 | Kelly Sizing | Overlay | Low | High | **T1** |
| 6 | Cross-Platform Arb | 2-4% | High | Medium | **T2** |
| 7 | OBI Signal | Unknown | Medium | Medium | **T2** |
| 8 | VPIN Flow Toxicity | Regime filter | Medium | Medium | **T2** |
| 9 | Whale Following | Unknown | Medium | Medium | **T2** |
| 10 | Sentiment Momentum | Unknown | High | Medium | **T2** |
| 11 | Overreaction Reversion | Unknown | Low | Medium | **T2** |
| 12 | Model-vs-Market | 10pp+ | High | Medium | **T2** |
| 13 | Combinatorial Arb | 0.24% of profits | Very High | Low | **T3** |
| 14 | Bayesian Networks | Unknown | Very High | Low | **T3** |
| 15 | Horizon Effect | 5-10pp | Medium | Medium | **T3** |
| 16 | Latency Arb | Saturated | Very High | Low | **T3** |
| 17 | Event-Driven News | Variable | Medium | Medium | **T3** |

---

## Key Academic Papers Referenced

1. Becker, J. (2025). "The Microstructure of Wealth Transfer in Prediction Markets." 72.1M trades, Kalshi.
2. IMDEA Networks (2025). "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets." arxiv 2508.03474.
3. Le, N.A. (2025). "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics." arxiv 2602.19520.
4. Semantic Non-Fungibility paper (2025). "Violations of the Law of One Price in Prediction Markets." arxiv 2601.01706.
5. Kelly Criterion for Prediction Markets (2024). arxiv 2412.14144.
6. Whelan, K. (2025). "On Optimal Betting Strategies with Multiple Mutually Exclusive Outcomes." Bulletin of Economic Research.
7. Whelan, K. (2025). "The Economics of the Kalshi Prediction Market Con." UCD Working Paper.
8. Easley, Lopez de Prado, O'Hara (2012). "Flow Toxicity and Liquidity in a High Frequency World."
9. Restocchi et al. (2019). "Improving prediction market forecasts by detecting and correcting possible over-reaction to price movements." EJOR.
10. Tetlock, P. (2015). "Superforecasting." + Wharton research on crowd prediction systems.
11. Bollen & Mao (2011). Twitter mood predicts the stock market.
12. Sontag et al. (2012). "Probability and Asset Updating using Bayesian Networks for Combinatorial Prediction Markets." arxiv 1210.4900.

---

## Recommended Implementation Order

1. **Kelly Sizing (#5)** -- Apply immediately to all existing trades. Zero cost, pure improvement.
2. **Longshot Bias (#1)** -- Post maker orders at extreme prices in high-bias categories. Start with Entertainment and World Events.
3. **Political Underconfidence (#4)** -- Build simple poll aggregation model. Trade political markets with >10pp model-market divergence.
4. **Market Making (#2)** -- Requires CLOB API integration. Start with 1-2 liquid markets, earn spread + longshot premium.
5. **OBI Signal (#7)** -- Add as alpha overlay to market making. Monitor orderbook depth for directional lean.
6. **VPIN Filter (#8)** -- Use as regime filter: only make markets when VPIN is low.
7. **Overreaction Reversion (#11)** -- Simple to implement, backtest first with historical price data.
8. **Model-vs-Market (#12)** -- Build domain models as time permits. Start with your strongest domain knowledge.
