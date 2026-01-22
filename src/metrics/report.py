import pandas as pd
from typing import Union
from backtester.results import BacktestResult, WalkForwardResult
from .performance import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    cagr,
    volatility,
    win_rate,
)
from .turnover import annualized_turnover
from .cost_analysis import gross_vs_net_sharpe, cost_drag_bps


def generate_report(result: Union[BacktestResult, WalkForwardResult], capital: float = 1_000_000) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("BACKTEST REPORT")
    lines.append("=" * 50)

    if isinstance(result, WalkForwardResult):
        returns = result.combined_returns
        cumulative = result.combined_cumulative
        positions = result.combined_positions
        pnl = result.combined_pnl
        costs = result.combined_costs
        lines.append(f"Type: Walk-Forward ({result.n_folds} folds)")
    else:
        returns = result.daily_returns
        cumulative = result.cumulative_returns
        positions = result.positions
        pnl = result.net_pnl
        costs = result.costs
        lines.append("Type: Single Backtest")

    lines.append(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    lines.append(f"Days: {len(returns)}")
    lines.append("")

    # performance metrics
    lines.append("-" * 50)
    lines.append("PERFORMANCE")
    lines.append("-" * 50)
    lines.append(f"Total Return:     {cumulative.iloc[-1] - 1:>10.2%}")
    lines.append(f"CAGR:             {cagr(cumulative, len(returns)):>10.2%}")
    lines.append(f"Volatility:       {volatility(returns):>10.2%}")
    lines.append(f"Sharpe Ratio:     {sharpe_ratio(returns):>10.2f}")
    lines.append(f"Sortino Ratio:    {sortino_ratio(returns):>10.2f}")
    lines.append(f"Calmar Ratio:     {calmar_ratio(returns, cumulative):>10.2f}")
    lines.append(f"Max Drawdown:     {max_drawdown(cumulative):>10.2%}")
    lines.append(f"Win Rate:         {win_rate(returns):>10.2%}")
    lines.append("")

    # turnover
    lines.append("-" * 50)
    lines.append("TURNOVER")
    lines.append("-" * 50)
    weights = positions / capital
    ann_turnover = annualized_turnover(weights)
    lines.append(f"Annual Turnover:  {ann_turnover:>10.1f}x")
    lines.append("")

    # costs
    lines.append("-" * 50)
    lines.append("COSTS")
    lines.append("-" * 50)
    total_costs = costs.sum()
    lines.append(f"Total Costs:      ${total_costs:>10,.0f}")
    lines.append(f"Cost Drag (bps):  {cost_drag_bps(pnl + costs, costs, capital):>10.1f}")

    # gross vs net
    gross_ret = (pnl + costs) / capital
    net_ret = pnl / capital
    comparison = gross_vs_net_sharpe(gross_ret, net_ret)
    lines.append(f"Gross Sharpe:     {comparison['gross_sharpe']:>10.2f}")
    lines.append(f"Net Sharpe:       {comparison['net_sharpe']:>10.2f}")
    lines.append(f"Sharpe Decay:     {comparison['decay_pct']:>10.1f}%")
    lines.append("")

    # fold breakdown for walk-forward
    if isinstance(result, WalkForwardResult):
        lines.append("-" * 50)
        lines.append("FOLD BREAKDOWN")
        lines.append("-" * 50)
        for fold in result.folds:
            lines.append(
                f"Fold {fold.fold_id}: {fold.test_start.date()} to {fold.test_end.date()} "
                f"Return: {fold.test_return:>7.2%}"
            )
        lines.append("")
        lines.append(f"Mean Fold Return: {result.mean_fold_return:>10.2%}")
        lines.append(f"Std Fold Return:  {result.std_fold_return:>10.2%}")

    lines.append("=" * 50)
    return "\n".join(lines)


def print_report(result: Union[BacktestResult, WalkForwardResult], capital: float = 1_000_000):
    print(generate_report(result, capital))


def get_metrics_dict(result: Union[BacktestResult, WalkForwardResult], capital: float = 1_000_000) -> dict:
    if isinstance(result, WalkForwardResult):
        returns = result.combined_returns
        cumulative = result.combined_cumulative
        positions = result.combined_positions
        pnl = result.combined_pnl
        costs = result.combined_costs
    else:
        returns = result.daily_returns
        cumulative = result.cumulative_returns
        positions = result.positions
        pnl = result.net_pnl
        costs = result.costs

    weights = positions / capital

    return {
        "total_return": cumulative.iloc[-1] - 1,
        "cagr": cagr(cumulative, len(returns)),
        "volatility": volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns, cumulative),
        "max_drawdown": max_drawdown(cumulative),
        "win_rate": win_rate(returns),
        "annual_turnover": annualized_turnover(weights),
        "total_costs": costs.sum(),
        "total_pnl": pnl.sum(),
    }
