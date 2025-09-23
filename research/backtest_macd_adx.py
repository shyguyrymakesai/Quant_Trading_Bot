import os
import time
import math
import ccxt
import numpy as np
import pandas as pd

# Choose your indicator lib
USE_TALIB = False  # set True if you installed TA-Lib
if USE_TALIB:
    import talib
else:
    import pandas_ta as ta

from backtesting import Backtest, Strategy

# -----------------------------
# 1) DATA LOADER (CCXT, BTC/USDT, 1h)
# -----------------------------
def load_ccxt_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=2000, exchange="binance"):
    if exchange.lower() == "binance":
        ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    elif exchange.lower() == "coinbase":
        # Coinbase Advanced (usable in the US)
        ex = ccxt.coinbaseadvanced({"enableRateLimit": True})
    else:
        raise ValueError("Unsupported exchange")

    # Fetch in one call (simple). If you want more history, paginate here.
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    # Backtesting.py expects capitalized column names
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


# -----------------------------
# 2) STRATEGY: MACD + ADX + VOL TARGET
# -----------------------------
class MACD_ADX_Strategy(Strategy):
    # MACD params
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    # ADX params
    adx_len = 14
    adx_threshold = 25
    # Vol-target params (heuristic for 1h data)
    vol_lb = 20             # lookback (bars) for realized vol
    target_vol = 0.02       # target (per-bar-ish proxy) — will tune in sweep
    size_min = 0.0
    size_max = 1.0

    def init(self):
        price = self.data.Close

        if USE_TALIB:
            macd, macdsignal, macdhist = talib.MACD(
                price.values, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
            )
            self.macd_hist = self.I(lambda: macdhist, name="macd_hist")
            adx = talib.ADX(self.data.High.values, self.data.Low.values, price.values, timeperiod=self.adx_len)
            self.adx = self.I(lambda: adx, name="adx")
        else:
            macd_df = ta.macd(price, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            adx_df = ta.adx(high=self.data.High, low=self.data.Low, close=price, length=self.adx_len)
            self.macd_hist = self.I(lambda: macd_df.iloc[:, 2].values, name="macd_hist")  # MACDh
            self.adx = self.I(lambda: adx_df.iloc[:, 0].values, name="adx")  # ADX

        # Volatility estimate (realized)
        rets = price.pct_change().fillna(0.0)
        self.realized_vol = self.I(lambda: rets.rolling(self.vol_lb).std().values, name="realized_vol")

    def position_size_scale(self):
        vol = self.realized_vol[-1]
        if vol is None or np.isnan(vol) or vol <= 0:
            return 0.0
        scale = float(self.target_vol / vol)
        return float(np.clip(scale, self.size_min, self.size_max))

    def next(self):
        # (2) ADX filter for trend strength
        if self.adx[-1] <= self.adx_threshold:
            return

        # MACD histogram zero-cross for direction
        crossed_up = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        crossed_down = self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0

        size_scale = self.position_size_scale()
        if size_scale <= 0:
            return

        if crossed_up:
            self.position.close()
            self.buy(size=size_scale)
        elif crossed_down:
            self.position.close()
            self.sell(size=size_scale)


def run_backtest(
    symbol="BTC/USDT",
    timeframe="1h",
    exchange="binance",
    commission=0.001,   # (2) ~0.10% taker fee
):
    df = load_ccxt_ohlcv(symbol=symbol, timeframe=timeframe, limit=2000, exchange=exchange)
    print("Data shape:", df.shape)
    print("Date range:", df.index.min(), "to", df.index.max())

    bt = Backtest(
        df,
        MACD_ADX_Strategy,
        cash=10_000,                # backtest notional
        commission=commission,      # fees matter!
        trade_on_close=True,
        exclusive_orders=True
    )
    stats = bt.run()
    print("\n" + "="*50 + "\nBACKTEST RESULTS\n" + "="*50)
    print(stats)
    # bt.plot(open_browser=False)  # uncomment to visualize when running locally
    return stats


if __name__ == "__main__":
    run_backtest(symbol="BTC/USDT", timeframe="1h", exchange="binance", commission=0.0012)
