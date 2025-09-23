import pandas as pd
import numpy as np
import talib
from .config import settings


def compute_indicators(ohlcv: list):
    """ohlcv: list of [ts, open, high, low, close, volume]"""
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    # TA-Lib MACD returns (macd, signal, hist)
    macd, macd_signal, macd_hist = talib.MACD(
        df["close"].values.astype(float),
        fastperiod=settings.macd_fast,
        slowperiod=settings.macd_slow,
        signalperiod=settings.macd_signal,
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    adx = talib.ADX(
        df["high"].values.astype(float),
        df["low"].values.astype(float),
        df["close"].values.astype(float),
        timeperiod=settings.adx_len,
    )
    df["ADX"] = adx
    return df


def volatility_target_size(df: pd.DataFrame):
    returns = df["close"].pct_change()
    vol = returns.rolling(settings.vol_lookback).std()
    # simple inverse-vol scaling toward a target
    scale = (settings.vol_target / (vol.replace(0, np.nan))).clip(
        lower=settings.min_size, upper=settings.max_size
    )
    return scale.fillna(0.0)


def last_signal(df: pd.DataFrame):
    row = df.iloc[-1]
    prev = df.iloc[-2]
    adx_ok = row.get("ADX", 0) > settings.adx_threshold
    # MACD histogram cross as signal (TA-Lib column)
    macd_hist = row.get("MACD_hist")
    macd_hist_prev = prev.get("MACD_hist")
    if adx_ok and macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > 0 and macd_hist_prev <= 0:
            return "buy"
        if macd_hist < 0 and macd_hist_prev >= 0:
            return "sell"
    return "hold"
