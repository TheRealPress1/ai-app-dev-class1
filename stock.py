import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf

sb.set_theme()

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Downloads OHLCV data from yfinance, sets DatetimeIndex,
        triggers return calculation, and returns the enriched DataFrame."""
        data = yf.download(self.symbol, start=self.start, end=self.end)

        # Flatten multi-level columns yfinance returns (e.g. ('Close', 'AAPL') -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ensure index is a proper DatetimeIndex (tz-naive)
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.index.name = "Date"

        self.calc_returns(data)
        return data

    def calc_returns(self, df):
        """Adds 'Change' (close-to-close difference) and 'Instant_Return'
        (daily log return) columns to the DataFrame. Vectorized — no loops."""
        df["Change"] = df["Close"].diff()
        df["Instant_Return"] = np.log(df["Close"]).diff().round(4)

    def add_technical_indicators(self, windows=[20, 50]):
        """Adds Simple Moving Averages (SMA) for each window to self.data
        and plots the closing price alongside all SMAs."""
        for window in windows:
            self.data[f"SMA_{window}"] = self.data["Close"].rolling(window).mean()

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(self.data.index, self.data["Close"], label="Close", linewidth=1.5)

        for window in windows:
            ax.plot(self.data.index, self.data[f"SMA_{window}"],
                    label=f"SMA {window}", linewidth=1.2, linestyle="--")

        ax.set_title(f"{self.symbol} — Closing Price & Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0f"))
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_return_dist(self):
        """Plots a histogram of the daily instantaneous (log) returns."""
        returns = self.data["Instant_Return"].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns, bins=50, edgecolor="white")
        ax.axvline(returns.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean: {returns.mean():.4f}")
        ax.set_title(f"{self.symbol} — Distribution of Daily Log Returns")
        ax.set_xlabel("Instantaneous Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """Plots cumulative growth of a $1 investment over the date range."""
        # Cumulative product of (1 + daily_return) starting from $1
        cumulative = (1 + self.data["Instant_Return"].dropna()).cumprod()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cumulative.index, cumulative, linewidth=1.5)
        ax.axhline(1, color="red", linestyle="--", linewidth=1, label="Break-even ($1)")
        ax.set_title(f"{self.symbol} — Cumulative Growth of $1 ({self.start} to {self.end})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value of $1 Investment")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.2f"))
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()


def main():
    # 1. Instantiate
    aapl = Stock("AAPL")

    # 2. Access the data attribute
    print(aapl.data.head())
    print(aapl.data.dtypes)

    # 3. Generate plots
    aapl.add_technical_indicators()
    aapl.plot_return_dist()
    aapl.plot_performance()


if __name__ == "__main__":
    main()
