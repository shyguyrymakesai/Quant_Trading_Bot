import pandas as pd
import numpy as np
import pandas_ta as ta
from quantbot.config import settings, get_symbol_params


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

    df["MACD"] = np.nan
    df["MACD_signal"] = np.nan
    df["MACD_hist"] = np.nan
    df["ADX"] = np.nan

    macd_df = ta.macd(
        df["close"], fast=int(mf), slow=int(ms), signal=int(sig)
    )
    if macd_df is not None and not macd_df.empty:
        macd_cols = macd_df.columns.tolist()
        if len(macd_cols) >= 3:
            df["MACD"] = macd_df[macd_cols[0]]
            df["MACD_signal"] = macd_df[macd_cols[1]]
            df["MACD_hist"] = macd_df[macd_cols[2]]

    adx_df = ta.adx(
        df["high"], df["low"], df["close"], length=int(adx_len)
    )
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if c.startswith("ADX")]
        if adx_cols:
            df["ADX"] = adx_df[adx_cols[0]]
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
    # MACD histogram cross as signal (pandas-ta column)
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
