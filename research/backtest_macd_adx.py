# Quick sanity-check backtest using Backtesting.py
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
import pandas as pd
import pandas_ta as ta

# NOTE: Replace GOOG with real crypto candles loaded via pandas
data = GOOG.copy()
data.rename(columns={'Close':'close','High':'high','Low':'low','Open':'open','Volume':'volume'}, inplace=True)

class MACD_ADX_Strategy(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_len = 14
    adx_threshold = 25

    def init(self):
        price = self.data.Close
        macd = ta.macd(price, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.macd_hist = self.I(lambda: macd['MACDh_12_26_9'].values, name='macd_hist')
        adx = ta.adx(high=self.data.High, low=self.data.Low, close=price, length=self.adx_len)
        self.adx = self.I(lambda: adx['ADX_14'].values, name='adx')

    def next(self):
        # Filter for trend strength
        if self.adx[-1] <= self.adx_threshold:
            return
        # MACD histogram cross
        if self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0:
            self.position.close()
            self.buy()
        elif self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0:
            self.position.close()
            self.sell()

bt = Backtest(data, MACD_ADX_Strategy, cash=10000, commission=.001, trade_on_close=True)
stats = bt.run()
print(stats)
# bt.plot()  # Uncomment to visualize
