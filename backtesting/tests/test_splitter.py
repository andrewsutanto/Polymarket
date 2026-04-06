"""Tests for train/test splitting — verify no temporal leakage."""

import pytest
from datetime import datetime, timedelta, timezone

from backtesting.data_loader import load_dataset
from backtesting.splitter import (
    simple_split, embargo_split, walk_forward_split,
    kfold_timeseries_split, get_splits,
)


@pytest.fixture
def dataset():
    return load_dataset(cities=["NYC"], days=90, eval_interval_min=60, seed=123)


class TestSimpleSplit:
    def test_no_overlap(self, dataset):
        split = simple_split(dataset, ratio=0.7)
        if split.train_data.n_bars > 0 and split.test_data.n_bars > 0:
            assert split.train_data.bars[-1].timestamp < split.test_data.bars[0].timestamp

    def test_covers_full_dataset(self, dataset):
        split = simple_split(dataset, ratio=0.7)
        total = split.train_data.n_bars + split.test_data.n_bars
        assert total == dataset.n_bars

    def test_ratio_approximately_correct(self, dataset):
        split = simple_split(dataset, ratio=0.7)
        actual_ratio = split.train_data.n_bars / dataset.n_bars
        assert abs(actual_ratio - 0.7) < 0.05


class TestEmbargoSplit:
    def test_embargo_gap_exists(self, dataset):
        split = embargo_split(dataset, ratio=0.7, embargo_days=3)
        if split.train_data.n_bars > 0 and split.test_data.n_bars > 0:
            gap = (split.test_data.bars[0].timestamp - split.train_data.bars[-1].timestamp)
            assert gap.total_seconds() > 3600 * 24 * 2  # At least 2 days

    def test_no_temporal_leakage(self, dataset):
        split = embargo_split(dataset, ratio=0.7, embargo_days=3)
        assert split.train_end <= split.test_start


class TestWalkForward:
    def test_no_overlap_between_windows(self, dataset):
        windows = walk_forward_split(dataset, train_days=30, test_days=10)
        assert len(windows) >= 1
        for w in windows:
            assert w.train_data.bars[-1].timestamp < w.test_data.bars[0].timestamp

    def test_windows_are_sequential(self, dataset):
        windows = walk_forward_split(dataset, train_days=30, test_days=10)
        for i in range(1, len(windows)):
            assert windows[i].test_start >= windows[i - 1].test_start


class TestKFoldTS:
    def test_expanding_training_set(self, dataset):
        folds = kfold_timeseries_split(dataset, n_folds=3)
        if len(folds) >= 2:
            assert folds[1].train_data.n_bars >= folds[0].train_data.n_bars

    def test_no_future_leakage(self, dataset):
        folds = kfold_timeseries_split(dataset, n_folds=3)
        for f in folds:
            if f.train_data.n_bars > 0 and f.test_data.n_bars > 0:
                assert f.train_data.bars[-1].timestamp < f.test_data.bars[0].timestamp

    def test_all_folds_start_from_beginning(self, dataset):
        folds = kfold_timeseries_split(dataset, n_folds=3)
        for f in folds:
            assert f.train_start == dataset.start_date


class TestGetSplits:
    def test_dispatch_simple(self, dataset):
        splits = get_splits(dataset, strategy="simple")
        assert len(splits) == 1

    def test_dispatch_walkforward(self, dataset):
        splits = get_splits(dataset, strategy="walkforward", train_days=30, test_days=10)
        assert len(splits) >= 1

    def test_invalid_strategy_raises(self, dataset):
        with pytest.raises(ValueError):
            get_splits(dataset, strategy="invalid")
