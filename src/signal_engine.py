import pandas as pd
import numpy as np
import talib
from .config import settings, get_symbol_params


def compute_indicators(ohlcv: list):
    """ohlcv: list of [ts, open, high, low, close, volume]"""
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    # Per-symbol overrides
    sym_params = get_symbol_params(settings.symbol)
    mf, ms, sig = sym_params.get(
        "macd", [settings.macd_fast, settings.macd_slow, settings.macd_signal]
    )
    adx_len = settings.adx_len
    # TA-Lib MACD returns (macd, signal, hist)
    macd, macd_signal, macd_hist = talib.MACD(
        df["close"].values.astype(float),
        fastperiod=int(mf),
        slowperiod=int(ms),
        signalperiod=int(sig),
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    adx = talib.ADX(
        df["high"].values.astype(float),
        df["low"].values.astype(float),
        df["close"].values.astype(float),
        timeperiod=int(adx_len),
    )
    df["ADX"] = adx
    return df


def volatility_target_size(df: pd.DataFrame):
    sym_params = get_symbol_params(settings.symbol)
    lb = int(sym_params.get("vol_lb", settings.vol_lookback))
    tgt = float(sym_params.get("target_vol", settings.vol_target))
    returns = df["close"].pct_change()
    vol = returns.rolling(lb).std()
    # simple inverse-vol scaling toward a target
    scale = (tgt / (vol.replace(0, np.nan))).clip(
        lower=settings.min_size, upper=settings.max_size
    )
    return scale.fillna(0.0)


def last_signal(df: pd.DataFrame):
    row = df.iloc[-1]
    prev = df.iloc[-2]
    sym_params = get_symbol_params(settings.symbol)
    adx_thr = int(sym_params.get("adx_threshold", settings.adx_threshold))
    adx_ok = row.get("ADX", 0) > adx_thr
    # MACD histogram cross as signal (TA-Lib column)
    macd_hist = row.get("MACD_hist")
    macd_hist_prev = prev.get("MACD_hist")
    if adx_ok and macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > 0 and macd_hist_prev <= 0:
            return "buy"
        if macd_hist < 0 and macd_hist_prev >= 0:
            return "sell"
    return "hold"


def compute_realized_vol(df: pd.DataFrame):
    sym_params = get_symbol_params(settings.symbol)
    lb = int(sym_params.get("vol_lb", settings.vol_lookback))
    returns = df["close"].pct_change()
    vol = returns.rolling(lb).std()
    if len(vol.dropna()) == 0:
        return 0.0
    return float(vol.iloc[-1])
