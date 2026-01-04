import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime


def generate_walk_forward_splits(
    dates: Union[pd.DatetimeIndex, List[datetime]],
    train_window: int,
    test_window: int,
    step: int = None,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate walk-forward train/test splits.

    Example with 252 day train, 63 day test, step 63:
    - Split 1: train days 0-251, test days 252-314
    - Split 2: train days 63-314, test days 315-377
    - etc.
    """
    if step is None:
        step = test_window

    dates = pd.DatetimeIndex(dates)
    n = len(dates)

    splits = []
    start = 0

    while start + train_window + test_window <= n:
        train_start = dates[start]
        train_end = dates[start + train_window - 1]
        test_start = dates[start + train_window]
        test_end_idx = min(start + train_window + test_window - 1, n - 1)
        test_end = dates[test_end_idx]

        splits.append((train_start, train_end, test_start, test_end))
        start += step

    return splits


def generate_expanding_splits(
    dates: Union[pd.DatetimeIndex, List[datetime]],
    min_train_window: int,
    test_window: int,
    step: int = None,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """
    Expanding window: train always starts from beginning, grows over time.
    """
    if step is None:
        step = test_window

    dates = pd.DatetimeIndex(dates)
    n = len(dates)

    splits = []
    train_end_idx = min_train_window - 1

    while train_end_idx + test_window < n:
        train_start = dates[0]
        train_end = dates[train_end_idx]
        test_start = dates[train_end_idx + 1]
        test_end_idx = min(train_end_idx + test_window, n - 1)
        test_end = dates[test_end_idx]

        splits.append((train_start, train_end, test_start, test_end))
        train_end_idx += step

    return splits


def split_data(
    data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[train_start:train_end]
    test = data.loc[test_start:test_end]
    return train, test
