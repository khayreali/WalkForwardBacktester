import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategies import MeanReversionStrategy, MomentumStrategy, weights_from_signals


class TestMeanReversion:
    def test_negative_signal_when_above_mean(self):
        # price trends up then stays high
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series([100 + i * 0.5 for i in range(50)], index=dates)
        returns = prices.pct_change().dropna()
        returns = returns.to_frame("TEST")

        strategy = MeanReversionStrategy(lookback=20)
        signals = strategy.generate_signals(returns)

        # after price has been rising, signal should be negative (sell)
        last_signal = signals["TEST"].iloc[-1]
        assert last_signal < 0, f"Expected negative signal, got {last_signal}"

    def test_positive_signal_when_below_mean(self):
        # price trends down
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series([100 - i * 0.5 for i in range(50)], index=dates)
        returns = prices.pct_change().dropna()
        returns = returns.to_frame("TEST")

        strategy = MeanReversionStrategy(lookback=20)
        signals = strategy.generate_signals(returns)

        last_signal = signals["TEST"].iloc[-1]
        assert last_signal > 0, f"Expected positive signal, got {last_signal}"

    def test_signals_clipped(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # extreme moves
        returns = pd.DataFrame({
            "A": np.random.randn(100) * 0.1,
            "B": np.random.randn(100) * 0.1,
        }, index=dates)

        strategy = MeanReversionStrategy(lookback=20)
        signals = strategy.generate_signals(returns)

        assert signals.max().max() <= 1.0
        assert signals.min().min() >= -1.0


class TestMomentum:
    def test_positive_signal_after_up_period(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # consistent positive returns
        returns = pd.DataFrame({
            "WINNER": [0.01] * 100,
            "LOSER": [-0.01] * 100,
        }, index=dates)

        strategy = MomentumStrategy(lookback=60, normalize="rank")
        signals = strategy.generate_signals(returns)

        # winner should have positive signal, loser negative
        last_winner = signals["WINNER"].iloc[-1]
        last_loser = signals["LOSER"].iloc[-1]

        assert last_winner > last_loser

    def test_cross_sectional_normalization(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.DataFrame({
            "A": [0.01] * 100,
            "B": [0.005] * 100,
            "C": [-0.005] * 100,
            "D": [-0.01] * 100,
        }, index=dates)

        strategy = MomentumStrategy(lookback=60, normalize="rank")
        signals = strategy.generate_signals(returns)

        last = signals.iloc[-1]
        # rank normalization: top performer should be 1, bottom should be -1
        assert last["A"] == 1.0  # highest momentum
        assert last["D"] == -0.5  # lowest momentum (rank 1/4 -> 0.25 -> scaled to -0.5)


class TestWeights:
    def test_equal_weights_sum(self):
        signals = pd.DataFrame({
            "A": [0.5],
            "B": [0.3],
            "C": [-0.4],
            "D": [-0.2],
        }, index=[datetime(2020, 1, 1)])

        weights = weights_from_signals(signals, method="equal", target_gross=2.0)

        gross = weights.abs().sum(axis=1).iloc[0]
        assert abs(gross - 2.0) < 0.01

    def test_proportional_weights(self):
        signals = pd.DataFrame({
            "A": [1.0],
            "B": [-1.0],
        }, index=[datetime(2020, 1, 1)])

        weights = weights_from_signals(signals, method="proportional", target_gross=2.0)

        assert weights["A"].iloc[0] > 0
        assert weights["B"].iloc[0] < 0
        gross = weights.abs().sum(axis=1).iloc[0]
        assert abs(gross - 2.0) < 0.01

    def test_no_signal_no_weight(self):
        signals = pd.DataFrame({
            "A": [0.0],
            "B": [0.0],
        }, index=[datetime(2020, 1, 1)])

        weights = weights_from_signals(signals, method="equal")

        assert weights.abs().sum().sum() == 0
