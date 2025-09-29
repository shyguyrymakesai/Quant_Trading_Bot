# Quick sanity-check backtest using Backtesting.py (uses pandas-ta indicators)
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
from contextlib import contextmanager

# NOTE: Replace GOOG with real crypto candles loaded via pandas
data = GOOG.copy()
# Keep the original column names (Open, High, Low, Close, Volume) for backtesting.py


class MACD_ADX_Strategy(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_len = 14
    adx_threshold = 25

    def init(self):
        # Convert price data to numpy arrays for indicator calculations
        close = np.array(self.data.Close, dtype=float)
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)

        macd_df = ta.macd(
            pd.Series(close),
            fast=int(self.macd_fast),
            slow=int(self.macd_slow),
            signal=int(self.macd_signal),
        )
        if macd_df is not None and not macd_df.empty and macd_df.shape[1] >= 3:
            macd_hist = macd_df.iloc[:, 2].to_numpy()
        else:
            macd_hist = np.full_like(close, np.nan, dtype=float)

        adx_df = ta.adx(
            pd.Series(high), pd.Series(low), pd.Series(close), length=int(self.adx_len)
        )
        if adx_df is not None and not adx_df.empty:
            adx_cols = [c for c in adx_df.columns if c.startswith("ADX")]
            if adx_cols:
                adx = adx_df[adx_cols[0]].to_numpy()
            else:
                adx = np.full_like(close, np.nan, dtype=float)
        else:
            adx = np.full_like(close, np.nan, dtype=float)

        # Create indicators for backtesting framework
        self.macd_hist = self.I(lambda: macd_hist, name="macd_hist")
        self.adx = self.I(lambda: adx, name="adx")

    def next(self):
        # Skip if we don't have enough data or ADX is NaN
        if len(self.data) < max(self.macd_slow, self.adx_len) + 1:
            return
        if np.isnan(self.adx[-1]) or np.isnan(self.macd_hist[-1]):
            return

        # Filter for trend strength
        if self.adx[-1] <= self.adx_threshold:
            return

        # MACD histogram cross
        if self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0:
            if self.position:
                self.position.close()
            self.buy()
        elif self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0:
            if self.position:
                self.position.close()
            self.sell()


@contextmanager
def suppress_sortino_runtimewarnings():
    """Suppress 'divide by zero' RuntimeWarnings from backtesting Sortino calc.
    Keeps output clean when there are no negative returns in the sample.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"divide by zero encountered in scalar divide",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            yield


if __name__ == "__main__":
    print("Running MACD + ADX Strategy Backtest...")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    bt = Backtest(
        data, MACD_ADX_Strategy, cash=10000, commission=0.001, trade_on_close=True
    )
    with suppress_sortino_runtimewarnings():
        stats = bt.run()
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(stats)
    print("\nEnvironment setup successful! ✅")
    # bt.plot()  # Uncomment to visualize
