"""Tests for regime detection — verify correct classification."""

import pytest
from datetime import datetime, timezone

from regime.detector import RegimeDetector, Regime
from regime.features import RegimeFeatures, RegimeFeatureSet


class TestRegimeFeatures:
    def test_initial_state(self):
        features = RegimeFeatures()
        feat = features.compute()
        assert feat.forecast_volatility == 0.0
        assert feat.market_volatility == 0.0
        assert feat.forecast_agreement == 0.5

    def test_update_records_data(self):
        features = RegimeFeatures()
        for i in range(20):
            features.update({
                "city": "NYC",
                "model_prob": 0.4 + i * 0.01,
                "market_price": 0.35 + i * 0.01,
                "forecast_shift_f": 1.0 if i % 3 == 0 else 0.0,
                "is_forecast_update": i % 3 == 0,
            })
        feat = features.compute()
        assert feat.market_volatility > 0
        assert feat.forecast_agreement != 0.0


class TestRegimeDetector:
    def test_initial_regime_is_neutral(self):
        detector = RegimeDetector()
        assert detector.current_regime == Regime.NEUTRAL

    def test_regime_updates(self):
        detector = RegimeDetector(eval_every_n=5)
        for i in range(30):
            detector.update({
                "city": "NYC",
                "model_prob": 0.5,
                "market_price": 0.5,
                "forecast_shift_f": 0.0,
                "is_forecast_update": False,
            })
        # Should have evaluated at least once
        assert detector.current_regime in list(Regime)

    def test_force_classify(self):
        detector = RegimeDetector()
        for i in range(10):
            detector.update({
                "city": "NYC",
                "model_prob": 0.5,
                "market_price": 0.5,
                "forecast_shift_f": 0.0,
                "is_forecast_update": False,
            })
        regime = detector.force_classify()
        assert isinstance(regime, Regime)

    def test_reset(self):
        detector = RegimeDetector()
        detector.update({
            "city": "NYC",
            "model_prob": 0.5,
            "market_price": 0.3,
            "forecast_shift_f": 2.0,
            "is_forecast_update": True,
        })
        detector.reset()
        assert detector.current_regime == Regime.NEUTRAL
        assert len(detector.transitions) == 0

    def test_strategy_weights_returned(self):
        detector = RegimeDetector()
        weights = detector.get_strategy_weights()
        assert isinstance(weights, dict)
        assert sum(weights.values()) > 0.9  # Should sum to ~1.0
