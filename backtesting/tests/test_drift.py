"""Tests for drift detection — verify alerts fire at correct thresholds."""

import pytest

from monitoring.drift_detector import DriftDetector, DriftBenchmarks, AlertLevel


class TestDriftDetector:
    def test_no_alerts_when_matching_benchmark(self):
        benchmarks = DriftBenchmarks(sharpe=1.0, win_rate=0.6, avg_edge=0.05, max_drawdown=0.10)
        detector = DriftDetector(benchmarks=benchmarks, window_size=20)

        alerts = []
        for i in range(25):
            # Alternating wins/losses matching 60% win rate
            pnl = 0.05 if i % 5 != 0 else -0.03
            result = detector.record_trade(pnl, edge=0.05, equity=50.0 + i * 0.02)
            alerts.extend(result)

        # Should have few or no alerts
        critical_alerts = [a for a in alerts if a.level in (AlertLevel.CRITICAL, AlertLevel.HALT)]
        assert len(critical_alerts) == 0

    def test_warning_on_25pct_degradation(self):
        benchmarks = DriftBenchmarks(sharpe=2.0, win_rate=0.7, avg_edge=0.10, max_drawdown=0.05)
        detector = DriftDetector(benchmarks=benchmarks, window_size=15)

        alerts = []
        for i in range(20):
            # Poor win rate: ~40%
            pnl = 0.03 if i % 5 < 2 else -0.02
            result = detector.record_trade(pnl, edge=0.04, equity=50.0)
            alerts.extend(result)

        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warning_alerts) > 0

    def test_critical_reduces_position_scale(self):
        benchmarks = DriftBenchmarks(sharpe=2.0, win_rate=0.8, avg_edge=0.10, max_drawdown=0.05)
        detector = DriftDetector(benchmarks=benchmarks, window_size=15)

        for i in range(20):
            # Terrible performance: all losses
            detector.record_trade(-0.05, edge=0.01, equity=50.0 - i * 0.5)

        assert detector.position_scale < 1.0

    def test_halt_on_negative_sharpe(self):
        benchmarks = DriftBenchmarks(sharpe=1.0, win_rate=0.6, avg_edge=0.05, max_drawdown=0.10)
        detector = DriftDetector(benchmarks=benchmarks, window_size=15)

        all_alerts = []
        for i in range(20):
            result = detector.record_trade(-0.10, edge=0.01, equity=50.0 - i * 2)
            all_alerts.extend(result)

        halt_alerts = [a for a in all_alerts if a.level == AlertLevel.HALT]
        assert len(halt_alerts) > 0

    def test_position_scale_zero_on_halt(self):
        benchmarks = DriftBenchmarks(sharpe=1.0, win_rate=0.6, avg_edge=0.05, max_drawdown=0.05)
        detector = DriftDetector(benchmarks=benchmarks, window_size=10)

        for i in range(15):
            detector.record_trade(-0.10, edge=0.01, equity=50.0 - i * 3)

        assert detector.position_scale == 0.0

    def test_callback_fires(self):
        benchmarks = DriftBenchmarks(sharpe=2.0, win_rate=0.8, avg_edge=0.10, max_drawdown=0.05)
        detector = DriftDetector(benchmarks=benchmarks, window_size=10)
        fired = []
        detector.on_alert(lambda a: fired.append(a))

        for i in range(15):
            detector.record_trade(-0.05, edge=0.01, equity=50.0 - i)

        assert len(fired) > 0
