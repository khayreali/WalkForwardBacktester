import pandas as pd
import numpy as np
from typing import Literal


def weights_from_signals(
    signals: pd.DataFrame,
    method: Literal["equal", "proportional"] = "equal",
    target_gross: float = 2.0,
) -> pd.DataFrame:
    """
    Convert signals to portfolio weights.
    target_gross: total gross exposure (sum of abs weights)
    """
    weights = pd.DataFrame(index=signals.index, columns=signals.columns, dtype=float)

    for idx in signals.index:
        row = signals.loc[idx].dropna()

        if len(row) == 0:
            weights.loc[idx] = 0
            continue

        if method == "equal":
            longs = row[row > 0].index
            shorts = row[row < 0].index

            w = pd.Series(0.0, index=row.index)
            if len(longs) > 0:
                w[longs] = 1.0 / len(longs)
            if len(shorts) > 0:
                w[shorts] = -1.0 / len(shorts)

        elif method == "proportional":
            w = row.copy()
            total = abs(w).sum()
            if total > 0:
                w = w / total * target_gross

        else:
            raise ValueError(f"Unknown method: {method}")

        # normalize to target gross exposure
        gross = abs(w).sum()
        if gross > 0:
            w = w * (target_gross / gross)

        weights.loc[idx, w.index] = w

    weights = weights.fillna(0)
    return weights


def normalize_weights(weights: pd.DataFrame, target_gross: float = 2.0) -> pd.DataFrame:
    result = weights.copy()
    for idx in result.index:
        row = result.loc[idx]
        gross = abs(row).sum()
        if gross > 0:
            result.loc[idx] = row * (target_gross / gross)
    return result


def long_only_weights(signals: pd.DataFrame, method: str = "equal") -> pd.DataFrame:
    # only take long positions
    long_signals = signals.clip(lower=0)
    weights = weights_from_signals(long_signals, method=method, target_gross=1.0)
    return weights
