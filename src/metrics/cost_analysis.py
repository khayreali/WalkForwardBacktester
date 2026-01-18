import pandas as pd
import numpy as np
from .performance import sharpe_ratio


def gross_vs_net_sharpe(gross_returns: pd.Series, net_returns: pd.Series) -> dict:
    gross_sr = sharpe_ratio(gross_returns)
    net_sr = sharpe_ratio(net_returns)
    return {
        "gross_sharpe": gross_sr,
        "net_sharpe": net_sr,
        "sharpe_decay": gross_sr - net_sr,
        "decay_pct": (gross_sr - net_sr) / gross_sr * 100 if gross_sr != 0 else 0,
    }


def cost_drag_bps(gross_pnl: pd.Series, costs: pd.Series, capital: float = 1_000_000) -> float:
    """Cost drag in basis points per year"""
    total_costs = costs.sum()
    days = len(costs)
    if days == 0:
        return 0.0
    annual_costs = total_costs * 252 / days
    return annual_costs / capital * 10000


def cost_per_trade(costs: pd.Series, trades: pd.DataFrame) -> float:
    """Average cost per trade"""
    n_trades = (trades.abs() > 0).sum().sum()
    if n_trades == 0:
        return 0.0
    return costs.sum() / n_trades


def breakeven_cost(gross_returns: pd.Series, target_sharpe: float = 0, periods: int = 252) -> float:
    """
    Cost (as fraction of daily pnl) that would reduce Sharpe to target.
    Returns as basis points.
    """
    current_sharpe = sharpe_ratio(gross_returns, periods=periods)
    if current_sharpe <= target_sharpe:
        return 0.0

    mean_ret = gross_returns.mean()
    std_ret = gross_returns.std()

    # sharpe = sqrt(252) * (mean - cost) / std = target
    # mean - cost = target * std / sqrt(252)
    # cost = mean - target * std / sqrt(252)
    required_mean = target_sharpe * std_ret / np.sqrt(periods)
    cost_per_day = mean_ret - required_mean

    # convert to bps (assuming 1.0 = 100%)
    return cost_per_day * 10000


def implementation_shortfall(
    gross_pnl: pd.Series,
    net_pnl: pd.Series,
    costs: pd.Series,
) -> dict:
    """Break down implementation shortfall"""
    total_gross = gross_pnl.sum()
    total_net = net_pnl.sum()
    total_costs = costs.sum()
    slippage = total_gross - total_net - total_costs

    return {
        "gross_pnl": total_gross,
        "explicit_costs": total_costs,
        "slippage": slippage,
        "net_pnl": total_net,
        "shortfall_pct": (total_gross - total_net) / abs(total_gross) * 100 if total_gross != 0 else 0,
    }
