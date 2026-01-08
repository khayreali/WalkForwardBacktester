import pandas as pd
import numpy as np
from typing import Optional
from .base import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, lookback: int = 60, normalize: str = "zscore", name: str = "momentum"):
        super().__init__(name)
        self.lookback = lookback
        self.normalize = normalize  # 'zscore', 'rank', or 'none'
        self.best_lookback = None

    def fit(self, train_returns: pd.DataFrame, lookback_range: Optional[range] = None) -> "MomentumStrategy":
        if lookback_range is None:
            self.fitted = True
            return self

        best_ic = -np.inf
        best_lb = self.lookback

        for lb in lookback_range:
            signals = self._compute_signals(train_returns, lb)
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
        return self._compute_signals(returns, self.lookback)

    def _compute_signals(self, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
        # cumulative return over lookback period
        rolling_ret = returns.rolling(lookback).sum()

        if self.normalize == "zscore":
            # cross-sectional z-score
            mean = rolling_ret.mean(axis=1)
            std = rolling_ret.std(axis=1)
            signals = rolling_ret.sub(mean, axis=0).div(std, axis=0)
        elif self.normalize == "rank":
            # cross-sectional rank scaled to [-1, 1]
            signals = rolling_ret.rank(axis=1, pct=True) * 2 - 1
        else:
            signals = rolling_ret

        signals = signals.clip(-1, 1)
        return signals
