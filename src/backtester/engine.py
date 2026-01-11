import pandas as pd
import numpy as np
from typing import Optional
from .results import BacktestResult


class Backtester:
    def __init__(
        self,
        strategy,
        cost_model=None,
        initial_capital: float = 1_000_000,
    ):
        self.strategy = strategy
        self.cost_model = cost_model
        self.initial_capital = initial_capital

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        signals: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        # generate signals if not provided
        if signals is None:
            signals = self.strategy.generate_signals(returns)

        # convert signals to weights
        weights = self.strategy.generate_weights(signals)

        # align everything
        common_idx = prices.index.intersection(returns.index).intersection(weights.index)
        prices = prices.loc[common_idx]
        returns = returns.loc[common_idx]
        weights = weights.loc[common_idx]

        # compute positions (in dollars)
        positions = weights * self.initial_capital

        # compute trades (change in position)
        trades = positions.diff().fillna(positions.iloc[0])

        # compute daily pnl from returns
        # pnl = position_yesterday * return_today
        prev_positions = positions.shift(1).fillna(0)
        gross_pnl = (prev_positions * returns).sum(axis=1)

        # compute costs
        if self.cost_model is not None:
            costs = self._compute_costs(trades, prices)
        else:
            costs = pd.Series(0, index=trades.index)

        net_pnl = gross_pnl - costs

        # compute returns
        capital = pd.Series(self.initial_capital, index=positions.index)
        capital = capital + net_pnl.cumsum()

        daily_returns = net_pnl / capital.shift(1).fillna(self.initial_capital)
        cumulative_returns = (1 + daily_returns).cumprod()

        return BacktestResult(
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            positions=positions,
            trades=trades,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
        )

    def _compute_costs(self, trades: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        daily_costs = pd.Series(0.0, index=trades.index)

        for date in trades.index:
            day_trades = trades.loc[date]
            day_prices = prices.loc[date]

            for ticker in day_trades.index:
                trade_value = abs(day_trades[ticker])
                if trade_value > 0 and ticker in day_prices.index:
                    price = day_prices[ticker]
                    if pd.notna(price) and price > 0:
                        shares = trade_value / price
                        cost = self.cost_model.calculate(price, shares)
                        daily_costs[date] += cost

        return daily_costs

    def __repr__(self):
        return f"Backtester(strategy={self.strategy}, capital={self.initial_capital:,.0f})"
