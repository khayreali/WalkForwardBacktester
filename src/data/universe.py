import pandas as pd
from typing import List, Optional
from datetime import datetime


SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ABT", "CRM",
    "ACN", "DHR", "NKE", "TXN", "NEE", "PM", "UPS", "MS", "RTX", "HON",
    "ORCL", "IBM", "QCOM", "LOW", "GS", "CAT", "BA", "AMGN", "SBUX", "GE",
]

FUTURES_SAMPLE = [
    "ES=F",  # S&P 500 E-mini
    "NQ=F",  # Nasdaq E-mini
    "YM=F",  # Dow E-mini
    "CL=F",  # Crude Oil
    "GC=F",  # Gold
    "SI=F",  # Silver
    "ZB=F",  # 30-Year Treasury
    "ZN=F",  # 10-Year Treasury
]

ETF_SAMPLE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "SLV",
    "TLT", "IEF", "LQD", "HYG", "XLF", "XLE", "XLK", "XLV",
]


class Universe:
    def __init__(self, tickers: List[str]):
        self.tickers = list(tickers)
        self._volume_data = None
        self._price_data = None

    def set_data(self, prices: pd.DataFrame, volume: Optional[pd.DataFrame] = None):
        self._price_data = prices
        self._volume_data = volume

    def filter_by_liquidity(self, min_avg_volume: float, window: int = 20) -> "Universe":
        if self._volume_data is None:
            return self

        avg_vol = self._volume_data.rolling(window).mean()
        liquid_mask = avg_vol.iloc[-1] >= min_avg_volume
        liquid_tickers = [t for t in self.tickers if t in liquid_mask.index and liquid_mask[t]]

        return Universe(liquid_tickers)

    def filter_by_price(self, min_price: float = 5.0) -> "Universe":
        if self._price_data is None:
            return self

        last_prices = self._price_data.iloc[-1]
        valid_tickers = [t for t in self.tickers if t in last_prices.index and last_prices[t] >= min_price]

        return Universe(valid_tickers)

    def get_active_tickers(self, date: datetime) -> List[str]:
        if self._price_data is None:
            return self.tickers

        if isinstance(date, str):
            date = pd.Timestamp(date)

        # find closest date
        idx = self._price_data.index.get_indexer([date], method="ffill")[0]
        if idx < 0:
            return []

        row = self._price_data.iloc[idx]
        return [t for t in self.tickers if t in row.index and pd.notna(row[t])]

    def __len__(self):
        return len(self.tickers)

    def __iter__(self):
        return iter(self.tickers)

    def __repr__(self):
        return f"Universe({len(self.tickers)} tickers)"


def get_sp500_sample() -> Universe:
    return Universe(SP500_SAMPLE)


def get_futures_sample() -> Universe:
    return Universe(FUTURES_SAMPLE)


def get_etf_sample() -> Universe:
    return Universe(ETF_SAMPLE)
