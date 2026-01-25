import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.performance import sharpe_ratio, sortino_ratio, max_drawdown, cagr, volatility
from metrics.turnover import calculate_turnover, annualized_turnover


class TestSharpeRatio:
    def test_known_sharpe(self):
        # daily return of 0.1% with 1% vol = sharpe ~1.58
        # mean = 0.001, std = 0.01
        # sharpe = sqrt(252) * 0.001 / 0.01 = 1.587
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)

        sr = sharpe_ratio(returns)
        # should be around 1.58 but with randomness
        assert 0.5 < sr < 3.0

    def test_zero_vol_zero_sharpe(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series([0.0] * 100, index=dates)

        sr = sharpe_ratio(returns)
        assert sr == 0

    def test_negative_returns_negative_sharpe(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = pd.Series([-0.001] * 252, index=dates)

        sr = sharpe_ratio(returns)
        assert sr < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        # always up
        cumret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        mdd = max_drawdown(cumret)
        assert mdd == 0

    def test_known_drawdown(self):
        # peak at 1.2, trough at 0.9 = -25%
        cumret = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9, 1.0])
        mdd = max_drawdown(cumret)
        assert abs(mdd - (-0.25)) < 0.01

    def test_full_loss(self):
        cumret = pd.Series([1.0, 0.5, 0.1])
        mdd = max_drawdown(cumret)
        assert mdd == -0.9


class TestCAGR:
    def test_known_cagr(self):
        # double in 252 days = ~100% annual return
        cumret = pd.Series([1.0, 2.0])
        annual = cagr(cumret, days=252)
        assert abs(annual - 1.0) < 0.01

    def test_flat_returns(self):
        cumret = pd.Series([1.0, 1.0, 1.0])
        annual = cagr(cumret, days=252)
        assert annual == 0


class TestVolatility:
    def test_known_vol(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # 1% daily vol = ~15.87% annual vol
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 252), index=dates)

        vol = volatility(returns)
        # should be around 15-16%
        assert 0.10 < vol < 0.25


class TestTurnover:
    def test_no_turnover(self):
        weights = pd.DataFrame({
            "A": [0.5, 0.5, 0.5],
            "B": [0.5, 0.5, 0.5],
        })
        turnover = calculate_turnover(weights)

        # after first day, no turnover
        assert turnover.iloc[1] == 0
        assert turnover.iloc[2] == 0

    def test_full_turnover(self):
        # flip from long A to long B
        weights = pd.DataFrame({
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
        })
        turnover = calculate_turnover(weights)

        # second day: |0-1| + |1-0| = 2, / 2 = 1
        assert turnover.iloc[1] == 1.0

    def test_annualized_turnover(self):
        weights = pd.DataFrame({
            "A": [0.5, 0.4, 0.5, 0.4],
            "B": [0.5, 0.6, 0.5, 0.6],
        })
        ann = annualized_turnover(weights)

        # should be positive
        assert ann > 0
