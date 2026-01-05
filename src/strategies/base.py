import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class Strategy(ABC):
    def __init__(self, name: str = "strategy"):
        self.name = name
        self.fitted = False

    def fit(self, train_returns: pd.DataFrame) -> "Strategy":
        self.fitted = True
        return self

    @abstractmethod
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        pass

    def generate_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        # default: equal weight long/short
        weights = signals.copy()

        for idx in weights.index:
            row = weights.loc[idx]
            longs = row[row > 0]
            shorts = row[row < 0]

            if len(longs) > 0:
                weights.loc[idx, longs.index] = 1.0 / len(longs)
            if len(shorts) > 0:
                weights.loc[idx, shorts.index] = -1.0 / len(shorts)

            # zero out neutrals
            weights.loc[idx, row == 0] = 0

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
