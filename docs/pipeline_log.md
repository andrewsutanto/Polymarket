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
