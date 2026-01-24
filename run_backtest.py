"""
Walk-forward backtest runner.
Compares mean reversion vs momentum strategies on SP500 stocks.
"""
import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime

from data import load_prices, load_returns, SP500_SAMPLE
from strategies import MeanReversionStrategy, MomentumStrategy
from backtester import WalkForwardBacktester
from costs import SimpleCostModel
from metrics import (
    sharpe_ratio,
    max_drawdown,
    annualized_turnover,
    print_report,
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_gross_vs_net,
    plot_fold_returns,
)

# config
TICKERS = SP500_SAMPLE
START = "2020-01-01"
END = "2024-12-31"
TRAIN_WINDOW = 252
TEST_WINDOW = 63
INITIAL_CAPITAL = 1_000_000

# cost model: 5bps spread + 1bps commission
COST_MODEL = SimpleCostModel(spread_bps=5, commission_bps=1)

# results dir
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("WALK-FORWARD BACKTESTER")
    print("=" * 60)
    print(f"Tickers: {len(TICKERS)} stocks from SP500")
    print(f"Period: {START} to {END}")
    print(f"Train window: {TRAIN_WINDOW} days")
    print(f"Test window: {TEST_WINDOW} days")
    print(f"Cost model: {COST_MODEL}")
    print()

    # load data
    print("Loading data...")
    prices = load_prices(TICKERS, START, END)
    returns = load_returns(TICKERS, START, END)
    print(f"Loaded {len(prices)} days of data for {len(prices.columns)} tickers")
    print()

    # strategies to test
    strategies = {
        "MeanRev": MeanReversionStrategy(lookback=20),
        "Momentum": MomentumStrategy(lookback=60),
    }

    results = {}
    metrics = {}

    for name, strategy in strategies.items():
        print("-" * 60)
        print(f"Running {name} strategy...")
        print("-" * 60)

        # run walk-forward backtest
        wf_backtester = WalkForwardBacktester(
            strategy=strategy,
            cost_model=COST_MODEL,
            train_window=TRAIN_WINDOW,
            test_window=TEST_WINDOW,
            initial_capital=INITIAL_CAPITAL,
        )

        result = wf_backtester.run(prices, returns)
        results[name] = result

        # calculate metrics
        net_sharpe = sharpe_ratio(result.combined_returns)

        # gross returns (add back costs)
        gross_pnl = result.combined_pnl + result.combined_costs
        gross_ret = gross_pnl / INITIAL_CAPITAL
        gross_sharpe = sharpe_ratio(gross_ret)

        mdd = max_drawdown(result.combined_cumulative)
        weights = result.combined_positions / INITIAL_CAPITAL
        turnover = annualized_turnover(weights)

        metrics[name] = {
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "max_dd": mdd,
            "turnover": turnover,
            "total_return": result.total_return,
            "n_folds": result.n_folds,
        }

        # print summary
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe (Gross): {gross_sharpe:.2f}")
        print(f"Sharpe (Net): {net_sharpe:.2f}")
        print(f"Max Drawdown: {mdd:.2%}")
        print(f"Annual Turnover: {turnover:.1f}x")
        print(f"Folds: {result.n_folds}")
        print()

        # generate plots
        plot_cumulative_returns(
            result,
            title=f"{name} - Cumulative Returns",
            save_path=RESULTS_DIR / f"{name.lower()}_cumulative.png",
        )

        plot_drawdown(
            result,
            title=f"{name} - Drawdown",
            save_path=RESULTS_DIR / f"{name.lower()}_drawdown.png",
        )

        plot_rolling_sharpe(
            result.combined_returns,
            title=f"{name} - Rolling Sharpe",
            save_path=RESULTS_DIR / f"{name.lower()}_rolling_sharpe.png",
        )

        plot_fold_returns(
            result,
            title=f"{name} - Fold Returns",
            save_path=RESULTS_DIR / f"{name.lower()}_folds.png",
        )

        # gross vs net
        gross_cumret = (1 + gross_ret).cumprod()
        plot_gross_vs_net(
            gross_cumret,
            result.combined_cumulative,
            title=f"{name} - Gross vs Net",
            save_path=RESULTS_DIR / f"{name.lower()}_gross_vs_net.png",
        )

        print(f"Plots saved to {RESULTS_DIR}/")
        print()

    # comparison table
    print("=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Strategy':<12} {'Sharpe (Gross)':<16} {'Sharpe (Net)':<14} {'Max DD':<10} {'Turnover':<10}")
    print("-" * 62)

    for name, m in metrics.items():
        max_dd_str = f"{m['max_dd']*100:.1f}%"
        turnover_str = f"{m['turnover']:.1f}x"
        print(
            f"{name:<12} "
            f"{m['gross_sharpe']:<16.2f} "
            f"{m['net_sharpe']:<14.2f} "
            f"{max_dd_str:<10} "
            f"{turnover_str:<10}"
        )

    print()
    print("=" * 60)

    # detailed reports
    print("\nDETAILED REPORTS")
    for name, result in results.items():
        print(f"\n{name}:")
        print_report(result, capital=INITIAL_CAPITAL)


if __name__ == "__main__":
    main()
