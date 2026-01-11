import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    gross_pnl: pd.Series
    costs: pd.Series
    net_pnl: pd.Series

    @property
    def total_return(self) -> float:
        return self.cumulative_returns.iloc[-1] - 1

    @property
    def total_pnl(self) -> float:
        return self.net_pnl.sum()

    @property
    def total_costs(self) -> float:
        return self.costs.sum()

    @property
    def n_trades(self) -> int:
        return (self.trades.abs() > 0).sum().sum()

    def __repr__(self):
        return (
            f"BacktestResult(total_return={self.total_return:.2%}, "
            f"total_pnl={self.total_pnl:,.0f}, "
            f"n_trades={self.n_trades})"
        )


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult

    @property
    def test_return(self) -> float:
        return self.result.total_return

    def __repr__(self):
        return (
            f"FoldResult(fold={self.fold_id}, "
            f"test={self.test_start.date()} to {self.test_end.date()}, "
            f"return={self.test_return:.2%})"
        )


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    combined_returns: pd.Series
    combined_cumulative: pd.Series
    combined_positions: pd.DataFrame
    combined_trades: pd.DataFrame
    combined_pnl: pd.Series
    combined_costs: pd.Series

    @property
    def total_return(self) -> float:
        return self.combined_cumulative.iloc[-1] - 1

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def fold_returns(self) -> List[float]:
        return [f.test_return for f in self.folds]

    @property
    def mean_fold_return(self) -> float:
        return np.mean(self.fold_returns)

    @property
    def std_fold_return(self) -> float:
        return np.std(self.fold_returns)

    def get_fold_summary(self) -> pd.DataFrame:
        data = []
        for f in self.folds:
            data.append({
                "fold": f.fold_id,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "return": f.test_return,
                "pnl": f.result.total_pnl,
                "costs": f.result.total_costs,
                "n_trades": f.result.n_trades,
            })
        return pd.DataFrame(data)

    def __repr__(self):
        return (
            f"WalkForwardResult(n_folds={self.n_folds}, "
            f"total_return={self.total_return:.2%}, "
            f"mean_fold_return={self.mean_fold_return:.2%})"
        )
