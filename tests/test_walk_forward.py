import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import generate_walk_forward_splits, generate_expanding_splits, split_data


class TestWalkForwardSplits:
    def test_correct_number_of_splits(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        train_window = 100
        test_window = 50
        step = 50

        splits = generate_walk_forward_splits(dates, train_window, test_window, step)

        # should have (500 - 100 - 50) / 50 + 1 = 8 splits
        expected = (500 - train_window - test_window) // step + 1
        assert len(splits) == expected

    def test_no_lookahead_bias(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        splits = generate_walk_forward_splits(dates, train_window=100, test_window=50)

        for train_start, train_end, test_start, test_end in splits:
            # train must end before test starts
            assert train_end < test_start
            # test must come after train
            assert test_start > train_start

    def test_train_test_no_overlap(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        splits = generate_walk_forward_splits(dates, train_window=100, test_window=50)

        for train_start, train_end, test_start, test_end in splits:
            # no overlap
            train_range = set(pd.date_range(train_start, train_end))
            test_range = set(pd.date_range(test_start, test_end))
            assert len(train_range & test_range) == 0

    def test_windows_correct_size(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        train_window = 100
        test_window = 50

        splits = generate_walk_forward_splits(dates, train_window, test_window)

        for train_start, train_end, test_start, test_end in splits:
            train_days = (train_end - train_start).days + 1
            assert train_days == train_window

            test_days = (test_end - test_start).days + 1
            assert test_days == test_window


class TestExpandingSplits:
    def test_train_starts_from_beginning(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        splits = generate_expanding_splits(dates, min_train_window=100, test_window=50)

        first_date = dates[0]
        for train_start, train_end, test_start, test_end in splits:
            assert train_start == first_date

    def test_train_window_expands(self):
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        splits = generate_expanding_splits(dates, min_train_window=100, test_window=50, step=50)

        train_sizes = []
        for train_start, train_end, test_start, test_end in splits:
            train_size = (train_end - train_start).days + 1
            train_sizes.append(train_size)

        # each subsequent train window should be larger or equal
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]


class TestSplitData:
    def test_correct_split(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"A": range(100)}, index=dates)

        train_start = dates[0]
        train_end = dates[49]
        test_start = dates[50]
        test_end = dates[99]

        train, test = split_data(data, train_start, train_end, test_start, test_end)

        assert len(train) == 50
        assert len(test) == 50
        assert train.index[0] == train_start
        assert test.index[-1] == test_end

    def test_no_overlap(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"A": range(100)}, index=dates)

        train, test = split_data(data, dates[0], dates[39], dates[40], dates[79])

        train_set = set(train.index)
        test_set = set(test.index)

        assert len(train_set & test_set) == 0
