import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from .engine import Backtester
from .results import BacktestResult, FoldResult, WalkForwardResult
from data.splits import generate_walk_forward_splits, split_data


class WalkForwardBacktester:
    def __init__(
        self,
        strategy,
        cost_model=None,
        train_window: int = 252,
        test_window: int = 63,
        step: Optional[int] = None,
        initial_capital: float = 1_000_000,
        expanding: bool = False,
    ):
        self.strategy = strategy
        self.cost_model = cost_model
        self.train_window = train_window
        self.test_window = test_window
        self.step = step if step is not None else test_window
        self.initial_capital = initial_capital
        self.expanding = expanding

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        fit_lookback_range: Optional[range] = None,
    ) -> WalkForwardResult:
        # generate splits
        if self.expanding:
            from data.splits import generate_expanding_splits
            splits = generate_expanding_splits(
                returns.index, self.train_window, self.test_window, self.step
            )
        else:
            splits = generate_walk_forward_splits(
                returns.index, self.train_window, self.test_window, self.step
            )

        if len(splits) == 0:
            raise ValueError("No valid splits generated. Check data length and window sizes.")

        folds = []
        all_returns = []
        all_positions = []
        all_trades = []
        all_pnl = []
        all_costs = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(splits):
            # split data
            train_returns, test_returns = split_data(returns, train_start, train_end, test_start, test_end)
            train_prices, test_prices = split_data(prices, train_start, train_end, test_start, test_end)

            # fit strategy on train data
            if fit_lookback_range is not None:
                self.strategy.fit(train_returns, lookback_range=fit_lookback_range)
            else:
                self.strategy.fit(train_returns)

            # generate signals on test data
            # need some lookback, so use data before test start
            lookback_start_idx = max(0, returns.index.get_loc(test_start) - 100)
            lookback_start = returns.index[lookback_start_idx]
            extended_returns = returns.loc[lookback_start:test_end]
            extended_prices = prices.loc[lookback_start:test_end]

            signals = self.strategy.generate_signals(extended_returns)
            signals = signals.loc[test_start:test_end]

            # run backtest on test period
            backtester = Backtester(
                strategy=self.strategy,
                cost_model=self.cost_model,
                initial_capital=self.initial_capital,
            )

            result = backtester.run(
                prices=test_prices,
                returns=test_returns,
                signals=signals,
            )

            fold = FoldResult(
                fold_id=fold_id,
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                result=result,
            )
            folds.append(fold)

            all_returns.append(result.daily_returns)
            all_positions.append(result.positions)
            all_trades.append(result.trades)
            all_pnl.append(result.net_pnl)
            all_costs.append(result.costs)

        # combine results
        combined_returns = pd.concat(all_returns)
        combined_positions = pd.concat(all_positions)
        combined_trades = pd.concat(all_trades)
        combined_pnl = pd.concat(all_pnl)
        combined_costs = pd.concat(all_costs)

        # handle overlapping dates (take last occurrence)
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep="last")]
        combined_positions = combined_positions[~combined_positions.index.duplicated(keep="last")]
        combined_trades = combined_trades[~combined_trades.index.duplicated(keep="last")]
        combined_pnl = combined_pnl[~combined_pnl.index.duplicated(keep="last")]
        combined_costs = combined_costs[~combined_costs.index.duplicated(keep="last")]

        combined_cumulative = (1 + combined_returns).cumprod()

        return WalkForwardResult(
            folds=folds,
            combined_returns=combined_returns,
            combined_cumulative=combined_cumulative,
            combined_positions=combined_positions,
            combined_trades=combined_trades,
            combined_pnl=combined_pnl,
            combined_costs=combined_costs,
        )

    def __repr__(self):
        mode = "expanding" if self.expanding else "rolling"
        return (
            f"WalkForwardBacktester({mode}, "
            f"train={self.train_window}, test={self.test_window}, step={self.step})"
        )
