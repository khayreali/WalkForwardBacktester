import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from backtester.results import BacktestResult, WalkForwardResult
from .performance import sharpe_ratio


def plot_cumulative_returns(
    result: Union[BacktestResult, WalkForwardResult],
    title: str = "Cumulative Returns",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    if isinstance(result, WalkForwardResult):
        cumret = result.combined_cumulative
    else:
        cumret = result.cumulative_returns

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumret.index, cumret.values, linewidth=1.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_drawdown(
    result: Union[BacktestResult, WalkForwardResult],
    title: str = "Drawdown",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    if isinstance(result, WalkForwardResult):
        cumret = result.combined_cumulative
    else:
        cumret = result.cumulative_returns

    peak = cumret.cummax()
    drawdown = (cumret - peak) / peak

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.3)
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3)
    ax.set_title(f"{title} ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_gross_vs_net(
    gross_cumret: pd.Series,
    net_cumret: pd.Series,
    title: str = "Gross vs Net Returns",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(gross_cumret.index, gross_cumret.values, label="Gross", linewidth=1.5)
    ax.plot(net_cumret.index, net_cumret.values, label="Net", linewidth=1.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_turnover_over_time(
    weights: pd.DataFrame,
    title: str = "Daily Turnover",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    weight_changes = weights.diff().fillna(weights.iloc[0])
    daily_turnover = weight_changes.abs().sum(axis=1) / 2

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(daily_turnover.index, daily_turnover.values, alpha=0.7, width=1)
    ax.axhline(y=daily_turnover.mean(), color="red", linestyle="--", label=f"Mean: {daily_turnover.mean():.2%}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_fold_returns(
    result: WalkForwardResult,
    title: str = "Returns by Fold",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
):
    fold_ids = [f.fold_id for f in result.folds]
    fold_returns = [f.test_return for f in result.folds]

    colors = ["green" if r > 0 else "red" for r in fold_returns]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(fold_ids, fold_returns, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=np.mean(fold_returns), color="blue", linestyle="--", label=f"Mean: {np.mean(fold_returns):.2%}")
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
