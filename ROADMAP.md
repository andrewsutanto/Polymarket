# Polymarket Bot Roadmap

**Last updated:** 2026-04-10
**Current state:** Paper trading functional, 6 strategies live, Phase 2 OOS results strong (82.4% WR, Sharpe 3.95 on 34 trades).

---

## Architecture Summary

The system has a clean layered design:
- **Data layer:** GammaFeed (market discovery) + CLOBFeed (order book polling) + FalconFeed (smart money/sentiment, optional)
- **Strategy layer:** 6 strategies behind `BaseStrategy` ABC, combined via `EnsembleStrategy` with regime-adaptive weights
- **Core models:** MarkovModel, BiasCalibrator, AlphaCombiner (IR = IC x sqrt(N))
- **Execution:** Executor with triple-flag live safety, paper mode with simulated slippage
- **Risk:** Kelly sizing, drawdown kill switch, per-cycle trade limits
- **Monitoring:** DriftDetector, HealthCheck, AlertManager (all wired to Telegram)
- **Infra:** SQLite via aiosqlite, Telegram UI (16 commands), dual entry points (main.py orchestrator, bot.py standalone)

---

## Tech Debt and Architectural Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| **Dual entry points** (main.py + bot.py + live_paper_trader.py) | Medium | Three separate orchestrators share logic but diverge. Consolidate to main.py + Telegram as the sole UI. |
| **Two Signal dataclasses** | High | `strategies.base.Signal` and `core.signal_engine.Signal` are different types. The signal_engine version references weather-specific fields (forecast_high_f, location, bucket_label). The strategies.base version is market-agnostic. main.py imports both. Unify or explicitly bridge them. |
| **Regime detector references weather-only strategies** | Medium | DEFAULT_REGIME_WEIGHTS keys are `forecast_arb`, `forecast_momentum`, `cross_city_arb` -- these do not match the actual strategy names in ALL_STRATEGIES (`implied_prob_arb`, `mean_reversion`, etc.). Regime weights are never applied correctly. |
| **Ensemble weights mismatch** | Medium | ensemble.py DEFAULT_WEIGHTS also reference `forecast_arb` etc., not the real strategy names. Dead code path. |
| **No order status tracking** | High | Live executor fires limit orders but has no fill confirmation loop, no partial fill handling, no cancel-replace logic. |
| **No position reconciliation** | High | Portfolio tracks state in-memory only. No on-chain balance check. Crash = lost state (SQLite write is async, may lag). |
| **No retry/circuit breaker on API calls** | Medium | GammaFeed, CLOBFeed, FalconFeed all use raw aiohttp with no exponential backoff or circuit breaker. |
| **SQLite not fit for production** | Low | Fine for paper trading at current scale. Will need Postgres or similar if running multiple instances or need concurrent reads. |

---

## Prioritized Roadmap

### Phase 3A: Go-Live Prerequisites (Weeks 1-3)

These must be done before risking real capital.

| # | Task | Why | Complexity | Role | Priority |
|---|------|-----|-----------|------|----------|
| 1 | **Fix Signal type unification + regime weight mapping** | Ensemble and regime weights reference nonexistent strategy names. Signals from strategies.base never reach the AlphaCombiner correctly. This means the entire signal combination pipeline is broken for the general-market bot. | S | Quant | P0 |
| 2 | **Live execution hardening** | Current `_execute_live` places a limit order and returns immediately with no fill confirmation. Need: (a) poll for fill status with timeout, (b) cancel stale orders, (c) handle partial fills, (d) implement order book walking for maker execution, (e) add nonce management for on-chain tx. | L | Data Engineer | P0 |
| 3 | **Position reconciliation with on-chain state** | On startup and periodically, query actual CLOB positions and USDC balance. Reconcile with in-memory Portfolio. Alert on any discrepancy. This prevents ghost positions after crashes. | M | Data Engineer | P0 |
| 4 | **Graceful shutdown + state persistence** | Ensure all open orders are cancelled on SIGTERM/SIGINT. Persist portfolio state to SQLite synchronously before exit. On restart, reload state. The current async write path can lose data. | S | Data Engineer | P0 |
| 5 | **Consolidate entry points** | Merge bot.py paper-trading logic into main.py. Eliminate live_paper_trader.py as a separate system. One bot, one state machine, two modes. | M | Data Engineer | P1 |

### Phase 3B: Alpha Generation (Weeks 2-5)

New signal sources, ordered by expected marginal Sharpe contribution.

| # | Task | Why | Complexity | Role | Priority |
|---|------|-----|-----------|------|----------|
| 6 | **Orderbook imbalance strategy** | CLOB data is already polled. Bid/ask depth imbalance at top-of-book is a strong short-term predictor (minutes to hours). High signal-to-noise on Polymarket because most LPs are unsophisticated. Requires: compute rolling imbalance ratio, backtest on stored book snapshots, add to strategy registry. | M | Quant | P1 |
| 7 | **Bayesian signal weight updater (connect AlphaCombiner to ensemble)** | AlphaCombiner already computes per-strategy IC and optimal weights via eigenvalue decomposition, but it is not wired into the EnsembleStrategy or the main loop. Connect it: after each resolved trade, update IC estimates and recompute weights. Replace static DEFAULT_WEIGHTS. | M | Quant | P1 |
| 8 | **Sentiment/news strategy via Falcon or public feeds** | FalconFeed integration exists but is disabled by default. Enable it, add a SentimentStrategy that acts on smart_money_bias and sentiment_score fields. Backtest with Falcon historical data if available, otherwise paper-trade for 2 weeks to calibrate thresholds. | M | Quant | P2 |
| 9 | **Temporal bias strategy (time-of-day / day-of-week)** | Prediction markets have known time-of-day patterns: US evening = retail flow, low liquidity. Compute hourly return profiles from historical CLOB data. Add as an overlay signal (boost/dampen confidence, not standalone). | S | Quant | P2 |
| 10 | **Smart money tracking** | Track top-N wallet addresses from on-chain Polymarket activity. Detect when known profitable wallets take positions. FalconFeed has `top_trader_agreement` but it is not used in any strategy. Wire it as a confirmation signal. | M | Quant | P2 |

### Phase 3C: Operational Readiness (Weeks 3-6)

| # | Task | Why | Complexity | Role | Priority |
|---|------|-----|-----------|------|----------|
| 11 | **Maker execution optimizer** | Current live executor uses market price as limit price. Should: (a) place orders inside spread to capture maker rebate, (b) walk the book if urgency is high, (c) implement time-in-force logic (IOC vs GTC). This is the difference between +2bps and -5bps per trade. | L | Data Engineer | P1 |
| 12 | **Docker deployment + .env management** | No Dockerfile exists. Create: multi-stage build, health check endpoint, .env.example with all required vars documented, docker-compose for bot + (optional) Postgres. | M | Data Engineer | P2 |
| 13 | **Performance analytics dashboard** | Current /pnl and /stats Telegram commands are basic. Add: (a) equity curve chart (matplotlib -> Telegram image), (b) per-strategy attribution, (c) rolling Sharpe, (d) trade-level breakdown exportable as CSV. Reuse infra/charts.py and infra/dashboard.py. | M | Telegram Engineer | P2 |
| 14 | **API rate limiting and circuit breakers** | Add exponential backoff + circuit breaker pattern to GammaFeed, CLOBFeed, and FalconFeed. Currently any sustained 429 or 5xx will crash the polling loop. Use `aiohttp-retry` or a simple decorator. | S | Data Engineer | P1 |
| 15 | **CI pipeline (GitHub Actions)** | Add: (a) pytest run on push, (b) mypy type checking, (c) ruff linting. Tests already exist in backtesting/tests/ and tests/. Just need the workflow YAML. | S | Data Engineer | P2 |

---

## Execution Order (Critical Path)

```
Week 1:  #1 (Signal unification) + #14 (Rate limiting) + #4 (Graceful shutdown)
Week 2:  #2 (Live execution hardening) + #6 (Orderbook imbalance strategy)
Week 3:  #3 (Position reconciliation) + #7 (Bayesian weight updater)
Week 4:  #5 (Consolidate entry points) + #11 (Maker optimizer) + #12 (Docker)
Week 5:  #8 (Sentiment strategy) + #13 (Analytics dashboard) + #15 (CI)
Week 6:  #9 (Temporal bias) + #10 (Smart money) — these are P2 enhancements
```

**Go-live gate:** Tasks 1-4 must pass before any real capital is deployed. Run paper mode for minimum 1 week after completing Phase 3A to validate the fixes under live market conditions.

---

## Key Metrics to Track Post-Launch

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Win rate (rolling 50) | > 65% | < 55% |
| Sharpe (rolling 30d) | > 2.0 | < 1.0 |
| Max drawdown | < 15% | > 10% triggers warning, > 20% triggers kill |
| Fill rate (live) | > 90% | < 70% |
| Avg slippage vs mid | < 1% | > 2% |
| API error rate | < 1% | > 5% |
| Signal-to-trade conversion | > 30% | < 15% |
