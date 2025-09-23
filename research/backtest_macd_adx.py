import numpy as np
import warnings
from contextlib import contextmanager
import pandas as pd
import ccxt
import yfinance as yf
from backtesting import Backtest, Strategy

# Use TA-Lib exclusively
USE_TALIB = True
import talib


def _parse_period_days(period: str | int | None) -> int:
    if isinstance(period, (int, float)):
        return int(period)
    if isinstance(period, str) and period.endswith("d"):
        num = period[:-1]
        if num.isdigit():
            return int(num)
    return 730


def _download_yf_hourly(symbol: str, total_days: int) -> pd.DataFrame:
    days = max(1, min(int(total_days), 720))
    df = yf.download(
        symbol,
        interval="1h",
        period=f"{days}d",
        progress=False,
        auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_df(symbol="BTC-USD", timeframe="1h", period="1825d", limit=1000):
    """Load OHLCV data from Coinbase Advanced with yfinance fallback."""
    try:
        ex = ccxt.coinbaseadvanced({"enableRateLimit": True, "timeout": 90000})
        raw = ex.fetch_ohlcv(symbol.replace("-", "/"), timeframe="1h", limit=limit)
        df = pd.DataFrame(
            raw, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
        df.set_index("Date", inplace=True)
        df = df.astype(float)
    except Exception as e:
        print(f"[WARN] Coinbase fetch failed ({e}); using yfinance {symbol}...")
        total_days = _parse_period_days(period)
        df = _download_yf_hourly(symbol, total_days)
        if df.empty:
            raise RuntimeError(f"No data from yfinance for {symbol}")
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    if timeframe == "4h":
        o = df["Open"].resample("4h").first()
        h = df["High"].resample("4h").max()
        l = df["Low"].resample("4h").min()
        c = df["Close"].resample("4h").last()
        v = df["Volume"].resample("4h").sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df.dropna()


def to_mbtc(df: pd.DataFrame) -> pd.DataFrame:
    """Scale USD/BTC prices → USD per mBTC so 1 unit = 0.001 BTC (integer units)."""
    out = df.copy()
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c] / 1_000.0
    return out


@contextmanager
def suppress_sortino_runtimewarnings():
    """Suppress 'divide by zero' RuntimeWarnings from backtesting Sortino calc.
    Keeps output clean when there are no negative day returns.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"divide by zero encountered in scalar divide",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            yield


class Strat(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_len = 14
    adx_threshold = 25
    vol_lb = 20
    target_vol = 0.02
    size_min = 0.0
    size_max = 1.0

    def init(self):
        price_series = pd.Series(self.data.Close).astype(float)
        high_series = pd.Series(self.data.High).astype(float)
        low_series = pd.Series(self.data.Low).astype(float)
        macd, macdsig, macdh = talib.MACD(
            price_series.values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        self.macd_hist = self.I(lambda: macdh, name="macd_hist")
        adx = talib.ADX(
            high_series.values,
            low_series.values,
            price_series.values,
            timeperiod=self.adx_len,
        )
        self.adx = self.I(lambda: adx, name="adx")

        rets = price_series.pct_change().fillna(0.0)
        self.realized_vol = self.I(
            lambda: rets.rolling(self.vol_lb).std().values, name="realized_vol"
        )

    def position_size_scale(self):
        vol = self.realized_vol[-1]
        if vol is None or np.isnan(vol) or vol <= 0:
            return 0.0
        return float(np.clip(self.target_vol / vol, self.size_min, self.size_max))

    def next(self):
        if len(self.macd_hist) < 2:
            return
        if (
            np.isnan(self.adx[-1])
            or np.isnan(self.macd_hist[-1])
            or np.isnan(self.macd_hist[-2])
        ):
            return
        if self.adx[-1] <= self.adx_threshold:
            return
        crossed_up = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        crossed_down = self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0
        vol_scale = self.position_size_scale()
        if vol_scale <= 0:
            return
        risk_frac = 0.10
        price = float(self.data.Close[-1])
        equity = float(self.equity)
        target_notional = max(0.0, equity * risk_frac * vol_scale)
        units = int(np.floor(target_notional / max(price, 1e-9)))
        if units < 1:
            units = 1
        if crossed_up:
            self.position.close()
            self.buy(size=units)
        elif crossed_down:
            self.position.close()
            self.sell(size=units)


def run_backtest(symbol="BTC-USD", timeframe="1h", commission=0.0012):
    df = load_df(symbol=symbol, timeframe=timeframe, period="1825d", limit=1500)
    df = to_mbtc(df)
    bt = Backtest(
        df,
        Strat,
        cash=10_000,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True,
    )
    with suppress_sortino_runtimewarnings():
        stats = bt.run()
    print(stats)
    return stats


if __name__ == "__main__":
    run_backtest(symbol="BTC-USD", timeframe="1h", commission=0.0012)
