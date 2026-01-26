# Walk-Forward Backtester

A walk-forward backtesting framework for evaluating trading strategies on US equities with realistic transaction cost modeling.

## Why Walk-Forward?

Traditional backtesting fits parameters on all available data and then tests on that same data. This leads to overfitting - strategies that look great in backtests but fail in live trading.

Walk-forward validation addresses this by:
1. Training on a historical window (e.g., 252 days)
2. Testing on the subsequent out-of-sample period (e.g., 63 days)
3. Rolling forward and repeating

This mimics how you'd actually use a strategy: train on past data, trade for a quarter, retrain, repeat. If a strategy works across multiple out-of-sample periods, there's more confidence it captures real market dynamics rather than noise.

## Why Costs Matter

Academic backtests often ignore transaction costs, but they're crucial for real trading. A strategy with 2.0 Sharpe before costs might have 0.5 Sharpe after costs - not worth trading. High-turnover strategies are particularly sensitive.

This framework tracks:
- Spread costs (half the bid-ask spread)
- Commission costs
- The resulting "Sharpe decay" from gross to net returns

## Strategies

### Mean Reversion
- Computes z-score of price relative to rolling mean
- Bets that prices revert to the mean
- Sells when price is high relative to recent history, buys when low
- Works best in range-bound markets

### Momentum
- Buys recent winners, sells recent losers
- Cross-sectionally normalized (rank or z-score across stocks)
- Works best when trends persist
- Higher turnover than mean reversion

## Features

- Walk-forward and expanding window validation
- Configurable train/test windows and step sizes
- Multiple cost models (simple, tiered)
- Performance metrics: Sharpe, Sortino, Max Drawdown, Calmar, CAGR
- Turnover analysis
- Gross vs net return comparison
- Visualization: cumulative returns, drawdowns, rolling Sharpe

## Installation

```bash
pip install -e .
```

Requires: numpy, pandas, matplotlib, yfinance, scipy, pytest

## Usage

Run the main backtest:

```bash
python run_backtest.py
```

This will:
1. Download SP500 sample data (2020-2024)
2. Run walk-forward backtests for mean reversion and momentum
3. Print performance comparison
4. Save plots to `results/`

## Sample Results

| Strategy | Sharpe (Gross) | Sharpe (Net) | Max DD | Turnover |
|----------|----------------|--------------|--------|----------|
| MeanRev  | X.XX           | X.XX         | XX.X%  | XX.Xx    |
| Momentum | X.XX           | X.XX         | XX.X%  | XX.Xx    |

*(Run `python run_backtest.py` to generate actual results)*

## Project Structure

```
├── src/
│   ├── data/           # Data loading, universe selection, splits
│   ├── strategies/     # Strategy base class, mean reversion, momentum
│   ├── backtester/     # Backtest engine, walk-forward runner
│   ├── costs/          # Transaction cost models
│   ├── metrics/        # Performance metrics, visualization
│   └── utils/          # Helpers
├── tests/              # Unit tests
├── data/               # Cached price data
├── results/            # Output plots and reports
├── notebooks/          # Analysis notebooks
└── run_backtest.py     # Main script
```

## Testing

```bash
pytest tests/ -v
```

## Would Add

Given more time, extensions would include:

- **More strategies**: Pairs trading, factor models, ML-based signals
- **Slippage modeling**: Volume-dependent market impact
- **Parameter optimization**: Grid search with cross-validation
- **Risk management**: Position limits, stop losses, volatility targeting
- **Multi-asset**: Futures, ETFs, cross-asset momentum
- **Regime detection**: Adjusting strategies based on market conditions
- **Execution simulation**: Limit orders, partial fills, queue position

## References

- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio
- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management
