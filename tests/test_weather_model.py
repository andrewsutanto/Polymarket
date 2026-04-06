"""Tests for the weather probability model."""

import pytest
from datetime import date, datetime, timezone

from core.weather_model import WeatherModel, BucketDef


@pytest.fixture
def model() -> WeatherModel:
    m = WeatherModel()
    buckets = [
        BucketDef("c70", "NYC", date(2026, 4, 10), 70, 71, "70-71°F"),
        BucketDef("c72", "NYC", date(2026, 4, 10), 72, 73, "72-73°F"),
        BucketDef("c74", "NYC", date(2026, 4, 10), 74, 75, "74-75°F"),
        BucketDef("c76", "NYC", date(2026, 4, 10), 76, 77, "76-77°F"),
        BucketDef("c78", "NYC", date(2026, 4, 10), 78, 79, "78-79°F"),
    ]
    m.register_buckets(buckets)
    return m


class TestBucketProbability:
    def test_probabilities_sum_to_one(self, model: WeatherModel) -> None:
        result = model.compute_probabilities(
            location="NYC",
            target_date=date(2026, 4, 10),
            forecast_high_f=74.0,
            lead_days=1,
            model_agreement=True,
        )
        assert result is not None
        total = sum(result.buckets.values())
        assert abs(total - 1.0) < 0.01

    def test_peak_bucket_is_forecast(self, model: WeatherModel) -> None:
        result = model.compute_probabilities(
            location="NYC",
            target_date=date(2026, 4, 10),
            forecast_high_f=74.5,
            lead_days=1,
            model_agreement=True,
        )
        assert result is not None
        peak_id = max(result.buckets, key=result.buckets.get)  # type: ignore
        assert peak_id == "c74"

    def test_disagreement_widens_sigma(self, model: WeatherModel) -> None:
        agreed = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 1, model_agreement=True
        )
        disagreed = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 1, model_agreement=False
        )
        assert agreed is not None and disagreed is not None
        assert disagreed.sigma_f > agreed.sigma_f

    def test_longer_lead_widens_sigma(self, model: WeatherModel) -> None:
        day1 = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 1, True
        )
        day4 = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 4, True
        )
        assert day1 is not None and day4 is not None
        assert day4.sigma_f > day1.sigma_f

    def test_no_buckets_returns_none(self, model: WeatherModel) -> None:
        result = model.compute_probabilities(
            location="Chicago",
            target_date=date(2026, 4, 10),
            forecast_high_f=65.0,
            lead_days=1,
            model_agreement=True,
        )
        assert result is None

    def test_stability_bonus(self, model: WeatherModel) -> None:
        # Record small shifts to increase stability
        for _ in range(5):
            model.record_shift("NYC", date(2026, 4, 10), 0.5)

        stable = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 1, True
        )

        model2 = WeatherModel()
        model2.register_buckets([
            BucketDef("c74", "NYC", date(2026, 4, 10), 74, 75, "74-75°F"),
        ])
        for _ in range(5):
            model2.record_shift("NYC", date(2026, 4, 10), 3.0)

        unstable = model2.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 1, True
        )

        assert stable is not None and unstable is not None
        assert stable.sigma_f < unstable.sigma_f


class TestBucketProbabilityMath:
    def test_extreme_forecast_concentrates_probability(self, model: WeatherModel) -> None:
        result = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.5, 0, True
        )
        assert result is not None
        assert result.buckets["c74"] > 0.4

    def test_all_probabilities_non_negative(self, model: WeatherModel) -> None:
        result = model.compute_probabilities(
            "NYC", date(2026, 4, 10), 74.0, 2, True
        )
        assert result is not None
        for p in result.buckets.values():
            assert p >= 0
