import pandas as pd
import numpy as np
from typing import Optional
from .base import Strategy


class MeanReversionStrategy(Strategy):
    def __init__(self, lookback: int = 20, name: str = "mean_reversion"):
        super().__init__(name)
        self.lookback = lookback
        self.best_lookback = None

    def fit(self, train_returns: pd.DataFrame, lookback_range: Optional[range] = None) -> "MeanReversionStrategy":
        if lookback_range is None:
            self.fitted = True
            return self

        # find optimal lookback by maximizing next-day return prediction
        prices = (1 + train_returns).cumprod()
        best_ic = -np.inf
        best_lb = self.lookback

        for lb in lookback_range:
            signals = self._compute_signals(prices, lb)
            # information coefficient: correlation of signal with next return
            next_ret = train_returns.shift(-1)
            ic = signals.corrwith(next_ret, axis=1).mean()
            if ic > best_ic:
                best_ic = ic
                best_lb = lb

        self.lookback = best_lb
        self.best_lookback = best_lb
        self.fitted = True
        return self

    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        prices = (1 + returns).cumprod()
        return self._compute_signals(prices, self.lookback)

    def _compute_signals(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        rolling_mean = prices.rolling(lookback).mean()
        rolling_std = prices.rolling(lookback).std()

        z_score = (prices - rolling_mean) / rolling_std

        # mean reversion: sell when high, buy when low
        signals = -z_score

        # clip to [-1, 1]
        signals = signals.clip(-1, 1)

        return signals
