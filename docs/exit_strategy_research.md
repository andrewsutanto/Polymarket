# Exit Strategy Research: Optimal Take-Profit, Stop-Loss, and Time Exits for Polymarket

## 1. Theoretical Foundation

### 1.1 Optimal Stopping Theory

The mathematical framework for exit decisions comes from **optimal stopping theory** -- choosing
the time to take an action to maximize expected reward. For binary outcome markets, this reduces
to: given a position with estimated edge, when does the expected value of exiting now exceed the
expected value of continuing to hold?

**Key insight from the literature**: For a binary contract that resolves to $0 or $1, the
hold-to-resolution EV is simply your estimated edge. Any early exit must be compared against this
baseline. Early exit is only justified when:
1. The risk-adjusted return of exiting exceeds hold-to-resolution, OR
2. Capital can be redeployed to a higher-edge opportunity (opportunity cost)

### 1.2 Kelly Criterion and Position Management

The Kelly criterion, applied to prediction markets (Meister, 2024 -- arXiv:2412.14144), shows:

- **Full Kelly** maximizes long-run geometric growth but creates a ~33% probability of halving
  bankroll before doubling it
- **Fractional Kelly** (0.25x-0.5x) is standard practice -- reduces volatility at cost of
  slower growth
- Binary contracts are bounded both above and below (unlike equities), making Kelly sizing
  inherently more conservative
- The inability to take naked short positions means trading two linked contracts (YES + NO = $1)

**Implication for exits**: Kelly assumes hold-to-resolution. Early exits reduce variance but also
reduce the geometric growth rate that Kelly optimizes. The mathematical optimum under Kelly is
to hold to resolution -- but this assumes perfect probability estimation.

### 1.3 Triple Barrier Method (Lopez de Prado, 2018)

The most rigorous framework for exit strategy in ML-based trading is the **Triple Barrier Method**:

1. **Upper barrier (take-profit)**: Fixed return above entry
2. **Lower barrier (stop-loss)**: Fixed loss below entry
3. **Vertical barrier (time exit)**: Maximum holding period

This framework labels trades as +1 (hit TP), -1 (hit SL), or 0 (time expired). It captures
the reality that a trade has direction, risk, AND timing constraints.

**For our Polymarket bot, this maps directly to the implementation.**


## 2. The Core Tradeoff: Hold-to-Resolution vs. Early Exit

### 2.1 Mathematical Analysis

**Scenario: Buy YES at $0.50, true probability = 55% (5% edge after fees)**

Hold to resolution:
- Win: 55% chance of +$0.50 profit per share
- Lose: 45% chance of -$0.50 loss per share
- EV = 0.55(0.50) - 0.45(0.50) = +$0.05 per share (10% return on capital)
- Variance = 0.55(0.50)^2 + 0.45(-0.50)^2 = 0.2475 - 0.0025 = 0.2475
- Std dev = $0.497 per share (massive relative to $0.05 EV)
- Sharpe (single trade) = 0.05 / 0.497 = 0.10

**The variance problem is severe**: You need ~100 independent trades at this edge to have
a 95% confidence of being profitable. Each individual trade is essentially a coin flip
with a tiny bias.

### 2.2 Early Exit Analysis

**Take-profit at +20% (price moves from $0.50 to $0.60):**

For the price to move from $0.50 to $0.60, the market must reprice the event as ~60% likely.
If true probability is 55%, this requires either:
- Market temporarily overshooting (momentum/news)
- Your edge being partially realized as new info emerges

Probability of reaching $0.60 before $0.40 (symmetric barriers):
- With true p=0.55, using gambler's ruin: P(reach 0.60 first) ~ 55-60%
- After ~2% round-trip fees: net profit = $0.10 - $0.02 = $0.08 per share
- Net loss if SL hit: -$0.10 - $0.02 = -$0.12 per share
- EV = 0.57(0.08) - 0.43(0.12) = 0.046 - 0.052 = -$0.006

**This is negative EV!** Symmetric TP/SL around a 5% edge position loses money because
fees eat the edge on the round trip.

**Take-profit at +30% (price to $0.65), stop-loss at -20% (price to $0.40):**
- P(reach 0.65 first | true p=0.55) ~ 45%
- Net TP profit: $0.15 - $0.02 = $0.13
- Net SL loss: -$0.10 - $0.02 = -$0.12
- EV = 0.45(0.13) - 0.55(0.12) = 0.059 - 0.066 = -$0.007

Still negative when fees are included. **The math is harsh: with 2% round-trip fees
and only 5% edge, most TP/SL combinations have negative EV.**

### 2.3 When Early Exit DOES Make Sense

1. **Edge has evaporated**: New information makes the position neutral or negative EV.
   Exit immediately regardless of P&L.

2. **Edge has been MORE than priced in**: Price moved significantly in your favor beyond
   what your model predicts. The market is now overpriced relative to your estimate.
   Example: bought at $0.50 (true prob 55%), price now at $0.70. Your model says 55%,
   market says 70%. Sell -- the position is now negative EV.

3. **Capital redeployment**: A better opportunity exists. If you have 3% edge on current
   position but see a 7% edge elsewhere, exit and redeploy.

4. **Risk of ruin**: Position is large relative to bankroll. Even with positive EV,
   the Kelly fraction may demand reducing exposure.

5. **Time decay near resolution**: As resolution approaches, if the event becomes clear
   (price moves to $0.90+), your remaining upside is small. Lock in gains.


## 3. Empirical Evidence: What Successful Polymarket Traders Do

### 3.1 Top Traders

**Theo4 / Fredi9999** ($22M+ profits):
- Strategy: Massive directional bets with private information advantage
- Commissioned custom polling data (YouGov "neighbor effect" survey)
- Hold to resolution -- their edge was from superior information, not market timing
- Lesson: If your edge is genuinely large (10%+), hold to resolution

**Arbitrage bots** ($40M+ extracted Apr 2024-Apr 2025):
- Strategy: Cross-market or YES/NO spread arbitrage
- Near-instant exit (seconds) -- hold time is minimal by design
- Average opportunity duration: 2.7 seconds (as of Q1 2026)
- Lesson: Different strategy class, not applicable to directional trading

### 3.2 Bot-Specific Exit Parameters (from open-source implementations)

Common configurations found across Polymarket bot codebases:
- **Stop-loss**: 15-20% of position value
- **Take-profit**: 25-30% of position value
- **Maximum hold time**: 7 days
- **Trailing stop**: 10-15% below peak price
- **Circuit breaker**: 40% total portfolio loss = halt all trading

### 3.3 Profitability Statistics

Only **7.6% of Polymarket wallets are profitable**. Of those, only 0.51% exceeded $1,000
in profits. This underscores the difficulty of the game and the importance of risk management.

Successful traders share three core traits:
1. Systematically capturing market pricing errors
2. Obsessive risk management
3. Patience to build information advantage in a specific domain


## 4. Polymarket-Specific Considerations

### 4.1 The Asymmetry Problem

Binary contracts create asymmetric payoffs depending on entry price:

| Entry Price | Max Gain (to $1) | Max Loss (to $0) | Gain/Loss Ratio |
|-------------|-------------------|-------------------|-----------------|
| $0.10       | +900%             | -100%             | 9:1             |
| $0.25       | +300%             | -100%             | 3:1             |
| $0.50       | +100%             | -100%             | 1:1             |
| $0.75       | +33%              | -100%             | 0.33:1          |
| $0.90       | +11%              | -100%             | 0.11:1          |

**Key insight**: Cheap contracts (< $0.30) have massive upside asymmetry -- holding to
resolution is much more attractive. Expensive contracts (> $0.70) have terrible risk/reward
for holding -- exits become more valuable.

### 4.2 Favorite-Longshot Bias

Research consistently finds that longshots are overpriced in prediction markets (people
over-bet unlikely outcomes). This means:

- **Buying cheap YES contracts ($0.05-$0.20)**: Likely negative edge on average due to
  longshot bias. Be very selective.
- **Buying expensive YES contracts ($0.80-$0.95)** or equivalently cheap NO contracts:
  Historically underpriced. Favorites are the better bet.
- **Exit implication**: On longshot positions, take profit more aggressively (they're
  more likely overpriced to begin with). On favorite positions, hold more patiently.

### 4.3 Near-Resolution Behavior (Last 24 Hours)

Markets converge rapidly as resolution approaches:
- Prices snap toward $0.95+ or $0.05 as the outcome becomes clear
- Liquidity often dries up in the final hours
- Remaining edge is minimal -- most alpha is captured well before resolution
- **Exit recommendation**: If position is profitable in the last 24 hours, hold.
  If position is losing, the loss is likely locked in -- cutting early saves nothing.

### 4.4 Liquidity Constraints

- Low-liquidity markets: Spread can be 5-10%, making exits expensive
- **Minimum trade size**: Positions should be >= $1.50 to guarantee exit capability
- Slippage on exit can exceed 2-3% in thin markets
- **Exit recommendation**: For illiquid markets, use wider stops (avoid getting
  stopped out by noise) and accept that exits may not execute at desired price.


## 5. Recommended Exit Strategy

### 5.1 Primary Recommendation: Model-Based Exits (Not Fixed Thresholds)

**The strongest finding from this research is that fixed TP/SL thresholds are suboptimal
for prediction markets.** Unlike equities/forex where price is continuous, prediction
market prices are bounded [0, 1] and reflect probability estimates. The correct exit
signal is: "has my edge changed?"

**Preferred exit logic:**
```
IF current_market_price > model_estimated_probability + fee_buffer:
    SELL (market has overpriced relative to your model)
IF model_estimated_probability has decreased (new info):
    SELL (your edge has evaporated)
IF better_opportunity_available AND capital_constrained:
    SELL (redeploy capital)
```

### 5.2 Backup: Fixed Triple-Barrier Thresholds

When model-based exits are not feasible (e.g., model is not real-time), use the
triple-barrier method with these thresholds:

#### Take-Profit Thresholds (by entry price tier)

| Entry Price  | Take-Profit | Rationale |
|-------------|-------------|-----------|
| $0.05-$0.20 | +50-80%     | Asymmetric upside; wide TP lets winners run |
| $0.20-$0.40 | +30-50%     | Good upside; moderate TP |
| $0.40-$0.60 | +20-30%     | Symmetric; tighter TP since downside equals upside |
| $0.60-$0.80 | +15-20%     | Limited upside; take profits quickly |
| $0.80-$0.95 | +8-12%      | Very limited upside; exit on any meaningful move |

#### Stop-Loss Thresholds (by entry price tier)

| Entry Price  | Stop-Loss   | Rationale |
|-------------|-------------|-----------|
| $0.05-$0.20 | -60-80%     | Wide stop -- these can fluctuate wildly; let edge play out |
| $0.20-$0.40 | -40-50%     | Moderate stop; avoid getting shaken out |
| $0.40-$0.60 | -25-35%     | Symmetric risk; reasonable stop |
| $0.60-$0.80 | -20-25%     | Tighter stop; limited upside means less room for loss |
| $0.80-$0.95 | -15-20%     | Tight stop; small edge means quick exit on adverse move |

#### Time Exit Thresholds

| Market Type             | Max Hold Time | Rationale |
|------------------------|---------------|-----------|
| Short-duration (< 7 days) | Until resolution | Not enough time for redeployment to matter |
| Medium-duration (7-30 days) | 7-14 days | Reassess if no movement after 1-2 weeks |
| Long-duration (> 30 days) | 14-21 days | Capital tied up too long; reassess and redeploy |

**Special case**: If within 48 hours of resolution, HOLD regardless of other signals
(unless stop-loss is hit). Near-resolution, the binary outcome dominates.

#### Trailing Stop

- **Trailing stop: 15% below peak price** (measured from highest price since entry)
- Only activates after position is at least +10% in profit (avoid premature triggering)
- Serves as a "lock in gains" mechanism when model-based exit is unavailable
- Wider trailing stop (20-25%) for cheap contracts ($0.05-$0.30)
- Tighter trailing stop (10-12%) for expensive contracts ($0.70-$0.95)

### 5.3 Decision Tree for the Bot

```
ON EACH PRICE UPDATE:
  1. [Model Check] Has the model's probability estimate changed?
     - If model_prob < market_price - 0.02: SELL (edge gone)
     - If model_prob > market_price + 0.05: HOLD (still good edge, don't exit early)

  2. [Take-Profit] Has price hit take-profit barrier?
     - Look up TP threshold based on entry price tier (table above)
     - If hit AND model says edge is gone: SELL
     - If hit BUT model still shows edge: HOLD (override TP -- model is king)

  3. [Stop-Loss] Has price hit stop-loss barrier?
     - Look up SL threshold based on entry price tier (table above)
     - If hit: SELL (hard stop, no override -- capital preservation is paramount)

  4. [Trailing Stop] Has price dropped 15% from peak?
     - If in profit AND trailing stop triggered: SELL
     - If not yet in profit: ignore trailing stop

  5. [Time Exit] Has max hold time elapsed?
     - If within 48 hours of resolution: HOLD (let it resolve)
     - If max hold time exceeded AND position is flat (< +5%): SELL and redeploy
     - If max hold time exceeded AND position is profitable: tighten trailing stop to 8%

  6. [Capital Redeployment] Is there a better opportunity?
     - If new opportunity has edge > 2x current position's remaining edge: SELL and redeploy
     - Otherwise: HOLD

  DEFAULT: HOLD
```

### 5.4 Anti-Patterns to Avoid

1. **Symmetric TP/SL with small edge**: A 10% TP / 10% SL with only 3-5% edge will lose
   money after fees. The math simply does not work.

2. **Tight stops on cheap contracts**: A $0.10 contract can easily fluctuate to $0.05
   and back. A 50% stop-loss would get triggered constantly on noise.

3. **No stop-loss at all**: While hold-to-resolution is EV-optimal, it creates enormous
   variance. A single bad run can deplete the bankroll before the law of large numbers
   kicks in. Always have a stop-loss, even if wide.

4. **Exiting profitable positions too early**: The asymmetry of binary markets means
   winners should run. A position that moved from $0.30 to $0.50 still has room to $1.00.
   Don't cut it at +66% if the model still shows edge.

5. **Holding losers hoping for recovery**: If the model's estimate has dropped (not just
   the market price), exit. New information that reduces your probability estimate is a
   legitimate reason to sell at a loss.


## 6. Implementation Constants

For initial deployment, use these constants (tune via backtesting):

```python
EXIT_CONFIG = {
    # Take-profit multipliers by entry price bucket
    "tp_multiplier": {
        (0.05, 0.20): 0.60,   # +60% from entry
        (0.20, 0.40): 0.40,   # +40%
        (0.40, 0.60): 0.25,   # +25%
        (0.60, 0.80): 0.18,   # +18%
        (0.80, 0.95): 0.10,   # +10%
    },
    # Stop-loss multipliers by entry price bucket
    "sl_multiplier": {
        (0.05, 0.20): -0.70,  # -70% from entry
        (0.20, 0.40): -0.45,  # -45%
        (0.40, 0.60): -0.30,  # -30%
        (0.60, 0.80): -0.22,  # -22%
        (0.80, 0.95): -0.18,  # -18%
    },
    # Trailing stop
    "trailing_stop_pct": 0.15,          # 15% below peak
    "trailing_stop_activation": 0.10,   # Only after +10% profit
    # Time exits
    "max_hold_days_short": None,        # Hold to resolution if < 7 days
    "max_hold_days_medium": 10,         # 10 days for 7-30 day markets
    "max_hold_days_long": 17,           # 17 days for 30+ day markets
    "near_resolution_hours": 48,        # Don't time-exit within 48h of resolution
    # Capital redeployment
    "redeploy_edge_ratio": 2.0,         # Redeploy if new edge > 2x current
    # Circuit breaker
    "max_portfolio_drawdown": 0.35,     # Halt at 35% total portfolio drawdown
    "max_daily_loss": 0.10,             # Halt at 10% daily loss
}
```


## 7. Summary of Key Findings

1. **Hold-to-resolution is mathematically optimal under Kelly** when your probability
   estimate is accurate. Fixed TP/SL thresholds reduce EV in exchange for variance
   reduction.

2. **Fees destroy early-exit profitability**: With ~2% round-trip fees and 3-5% edge,
   most TP/SL combinations have negative EV. Exits should be driven by model updates,
   not fixed price levels.

3. **Model-based exits dominate fixed thresholds**: The best exit signal is "my model
   no longer shows edge on this position."

4. **Entry price determines optimal thresholds**: Cheap contracts need wide stops and
   wide TP. Expensive contracts need tight stops and tight TP.

5. **Stop-loss is essential for survival**: Even though it reduces EV, it prevents
   ruin. The 7.6% profitability rate on Polymarket shows most traders fail to manage
   downside risk.

6. **Top traders hold to resolution**: Theo4/Fredi9999 made $22M+ by having genuinely
   superior information and holding through volatility. If your edge is real and large,
   holding is correct.

7. **The triple barrier method** (TP + SL + time) from Lopez de Prado provides the
   cleanest implementation framework for a trading bot.

8. **Trailing stops work for locking in gains** when you can't continuously re-evaluate
   your model. Set at 15% below peak, activated after +10% profit.


## Sources

- Meister, B.K. (2024). "Application of the Kelly Criterion to Prediction Markets." arXiv:2412.14144
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley. (Triple Barrier Method)
- Bawa, N. "The Math of Prediction Markets: Binary Options, Kelly Criterion, and CLOB Pricing Mechanics." Substack.
- Ottaviani, M. & Sorensen, P. "Noise, Information and the Favorite-Longshot Bias." IGIER Bocconi.
- MDPI (2024). "Trading Binary Options Using Expected Profit and Loss Metrics."
- Polymarket on-chain analysis: 95M transactions, 6 profit models (2025 report)
- QuestDB. "Optimal Stopping Theory in Trading Algorithms."
