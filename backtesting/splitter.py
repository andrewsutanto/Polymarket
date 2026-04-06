"""Train/test splitting strategies with strict temporal ordering.

Implements simple, walk-forward, k-fold time-series, and embargo splits.
No future data ever leaks into training sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backtesting.data_loader import BacktestDataset

logger = logging.getLogger(__name__)


@dataclass
class SplitWindow:
    """A single train/test split with temporal boundaries."""

    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: BacktestDataset
    test_data: BacktestDataset

    def __post_init__(self) -> None:
        assert self.train_end <= self.test_start, (
            f"Temporal leakage: train_end={self.train_end} > test_start={self.test_start}"
        )
        if self.train_data.n_bars > 0 and self.test_data.n_bars > 0:
            assert self.train_data.bars[-1].timestamp < self.test_data.bars[0].timestamp, (
                "Temporal leakage: last train bar >= first test bar"
            )


def simple_split(
    dataset: BacktestDataset,
    ratio: float = 0.7,
    cutoff_date: datetime | None = None,
) -> SplitWindow:
    """Split at a single cutoff point.

    Args:
        dataset: Full dataset.
        ratio: Fraction for training. Ignored if cutoff_date given.
        cutoff_date: Explicit cutoff datetime.

    Returns:
        Single SplitWindow.
    """
    if cutoff_date is None:
        n = dataset.n_bars
        cut_idx = int(n * ratio)
        cutoff_date = dataset.bars[cut_idx].timestamp

    train = dataset.slice(dataset.start_date, cutoff_date)
    test = dataset.slice(cutoff_date, dataset.end_date + timedelta(days=1))

    logger.info("Simple split: train=%d bars, test=%d bars", train.n_bars, test.n_bars)
    return SplitWindow(
        fold=0,
        train_start=dataset.start_date,
        train_end=cutoff_date,
        test_start=cutoff_date,
        test_end=dataset.end_date,
        train_data=train,
        test_data=test,
    )


def embargo_split(
    dataset: BacktestDataset,
    ratio: float = 0.7,
    embargo_days: int = 3,
) -> SplitWindow:
    """Split with a gap between train and test.

    Args:
        dataset: Full dataset.
        ratio: Fraction for training.
        embargo_days: Days to skip between train and test.

    Returns:
        Single SplitWindow.
    """
    n = dataset.n_bars
    cut_idx = int(n * ratio)
    train_end = dataset.bars[cut_idx].timestamp
    test_start = train_end + timedelta(days=embargo_days)

    train = dataset.slice(dataset.start_date, train_end)
    test = dataset.slice(test_start, dataset.end_date + timedelta(days=1))

    logger.info("Embargo split: train=%d, embargo=%dd, test=%d", train.n_bars, embargo_days, test.n_bars)
    return SplitWindow(
        fold=0,
        train_start=dataset.start_date,
        train_end=train_end,
        test_start=test_start,
        test_end=dataset.end_date,
        train_data=train,
        test_data=test,
    )


def walk_forward_split(
    dataset: BacktestDataset,
    train_days: int = 60,
    test_days: int = 15,
) -> list[SplitWindow]:
    """Rolling walk-forward splits.

    Args:
        dataset: Full dataset.
        train_days: Days in each training window.
        test_days: Days in each test window.

    Returns:
        List of SplitWindows.
    """
    windows: list[SplitWindow] = []
    cursor = dataset.start_date
    fold = 0

    while True:
        train_end = cursor + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)

        if test_end > dataset.end_date + timedelta(hours=1):
            break

        train = dataset.slice(cursor, train_end)
        test = dataset.slice(test_start, test_end)

        if train.n_bars < 10 or test.n_bars < 5:
            cursor += timedelta(days=test_days)
            continue

        windows.append(SplitWindow(
            fold=fold,
            train_start=cursor,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_data=train,
            test_data=test,
        ))
        fold += 1
        cursor += timedelta(days=test_days)

    logger.info("Walk-forward: %d windows (train=%dd, test=%dd)", len(windows), train_days, test_days)
    return windows


def kfold_timeseries_split(
    dataset: BacktestDataset,
    n_folds: int = 5,
) -> list[SplitWindow]:
    """Expanding-window time-series cross-validation.

    Fold k trains on all data up to period k, tests on period k+1.

    Args:
        dataset: Full dataset.
        n_folds: Number of test folds.

    Returns:
        List of SplitWindows with expanding training sets.
    """
    total_days = (dataset.end_date - dataset.start_date).days
    fold_days = total_days / (n_folds + 1)

    windows: list[SplitWindow] = []
    for k in range(n_folds):
        train_end = dataset.start_date + timedelta(days=fold_days * (k + 1))
        test_start = train_end
        test_end = min(train_end + timedelta(days=fold_days), dataset.end_date + timedelta(hours=1))

        train = dataset.slice(dataset.start_date, train_end)
        test = dataset.slice(test_start, test_end)

        if train.n_bars < 10 or test.n_bars < 5:
            continue

        windows.append(SplitWindow(
            fold=k,
            train_start=dataset.start_date,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_data=train,
            test_data=test,
        ))

    logger.info("K-fold TS: %d folds (expanding window)", len(windows))
    return windows


def get_splits(
    dataset: BacktestDataset,
    strategy: str = "simple",
    ratio: float = 0.7,
    embargo_days: int = 3,
    train_days: int = 60,
    test_days: int = 15,
    n_folds: int = 5,
    cutoff_date: datetime | None = None,
) -> list[SplitWindow]:
    """Dispatch to the correct split strategy."""
    if strategy == "simple":
        return [simple_split(dataset, ratio, cutoff_date)]
    elif strategy == "embargo":
        return [embargo_split(dataset, ratio, embargo_days)]
    elif strategy == "walkforward":
        return walk_forward_split(dataset, train_days, test_days)
    elif strategy == "kfold":
        return kfold_timeseries_split(dataset, n_folds)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
