import pandas as pd
import numpy as np


def sharpe_ratio(returns: pd.Series, rf: float = 0, periods: int = 252) -> float:
    excess = returns - rf / periods
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, rf: float = 0, periods: int = 252) -> float:
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    return np.sqrt(periods) * excess.mean() / downside_std


def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def max_drawdown_duration(cumulative_returns: pd.Series) -> int:
    peak = cumulative_returns.cummax()
    underwater = cumulative_returns < peak

    if not underwater.any():
        return 0

    # find drawdown periods
    starts = (~underwater).shift(1, fill_value=True) & underwater
    ends = underwater & (~underwater).shift(-1, fill_value=True)

    durations = []
    start_idx = None

    for i, date in enumerate(cumulative_returns.index):
        if starts.iloc[i]:
            start_idx = i
        if ends.iloc[i] and start_idx is not None:
            durations.append(i - start_idx + 1)
            start_idx = None

    # if still in drawdown
    if start_idx is not None:
        durations.append(len(cumulative_returns) - start_idx)

    return max(durations) if durations else 0


def calmar_ratio(returns: pd.Series, cumulative_returns: pd.Series, periods: int = 252) -> float:
    ann_return = cagr(cumulative_returns, len(returns))
    mdd = abs(max_drawdown(cumulative_returns))
    if mdd == 0:
        return np.inf if ann_return > 0 else 0.0
    return ann_return / mdd


def cagr(cumulative_returns: pd.Series, days: int, periods: int = 252) -> float:
    if days == 0:
        return 0.0
    total_return = cumulative_returns.iloc[-1]
    years = days / periods
    if years == 0:
        return 0.0
    return total_return ** (1 / years) - 1


def total_return(cumulative_returns: pd.Series) -> float:
    return cumulative_returns.iloc[-1] - 1


def volatility(returns: pd.Series, periods: int = 252) -> float:
    return returns.std() * np.sqrt(periods)


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses
