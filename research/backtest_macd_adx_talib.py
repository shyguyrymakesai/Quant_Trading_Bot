# Quick sanity-check backtest using Backtesting.py (Modified to use TA-Lib instead of pandas-ta)
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
import pandas as pd
import talib
import numpy as np

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
        # Convert price data to numpy arrays for TA-Lib
        close = np.array(self.data.Close, dtype=float)
        high = np.array(self.data.High, dtype=float)
        low = np.array(self.data.Low, dtype=float)

        # Calculate MACD using TA-Lib
        macd_line, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # Calculate ADX using TA-Lib
        adx = talib.ADX(high, low, close, timeperiod=self.adx_len)

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


if __name__ == "__main__":
    print("Running MACD + ADX Strategy Backtest...")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    bt = Backtest(
        data, MACD_ADX_Strategy, cash=10000, commission=0.001, trade_on_close=True
    )
    stats = bt.run()
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(stats)
    print("\nEnvironment setup successful! ✅")
    # bt.plot()  # Uncomment to visualize
