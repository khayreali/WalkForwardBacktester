import pandas as pd
import numpy as np


def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover = sum of absolute weight changes / 2"""
    weight_changes = weights.diff().fillna(weights.iloc[0])
    daily_turnover = weight_changes.abs().sum(axis=1) / 2
    return daily_turnover


def average_turnover(weights: pd.DataFrame) -> float:
    daily = calculate_turnover(weights)
    return daily.mean()


def annualized_turnover(weights: pd.DataFrame, periods: int = 252) -> float:
    return average_turnover(weights) * periods


def turnover_from_positions(positions: pd.DataFrame, capital: float) -> pd.Series:
    """Turnover from dollar positions"""
    weights = positions / capital
    return calculate_turnover(weights)


def trade_count(weights: pd.DataFrame, threshold: float = 0.001) -> int:
    """Count number of trades above threshold"""
    weight_changes = weights.diff().fillna(weights.iloc[0])
    return (weight_changes.abs() > threshold).sum().sum()


def average_holding_period(weights: pd.DataFrame) -> float:
    """Approximate holding period in days"""
    turnover = average_turnover(weights)
    if turnover == 0:
        return np.inf
    return 1 / turnover
