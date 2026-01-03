import os
import pandas as pd
import yfinance as yf
from typing import List, Optional
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def _get_cache_path(tickers: List[str], start: str, end: str, suffix: str) -> Path:
    ticker_str = "_".join(sorted(tickers)[:5])  # limit filename length
    if len(tickers) > 5:
        ticker_str += f"_plus{len(tickers)-5}"
    return CACHE_DIR / f"{ticker_str}_{start}_{end}_{suffix}.pkl"


def load_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_path = _get_cache_path(tickers, start, end, "prices")

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    data = yf.download(tickers, start=start, end=end, progress=False)

    if len(tickers) == 1:
        prices = data["Close"].to_frame(tickers[0])
    else:
        prices = data["Close"]

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(0)

    prices = prices.dropna(how="all")

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_pickle(cache_path)

    return prices


def load_returns(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_path = _get_cache_path(tickers, start, end, "returns")

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    prices = load_prices(tickers, start, end, use_cache=use_cache)
    returns = prices.pct_change().dropna()

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        returns.to_pickle(cache_path)

    return returns


def load_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_path = _get_cache_path(tickers, start, end, "ohlcv")

    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    data = yf.download(tickers, start=start, end=end, progress=False)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        data.to_pickle(cache_path)

    return data


def clear_cache():
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
