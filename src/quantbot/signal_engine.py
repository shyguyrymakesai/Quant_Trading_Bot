from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
from quantbot.config import settings, get_symbol_params


def _resolve_symbol(symbol: Optional[str]) -> str:
    """Return a symbol string falling back to global settings."""
    return symbol or getattr(settings, "current_symbol", None) or settings.symbol


def compute_indicators(ohlcv: list, *, symbol: Optional[str] = None):
    """ohlcv: list of [ts, open, high, low, close, volume]"""
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    # Per-symbol overrides
    eff_symbol = _resolve_symbol(symbol)
    sym_params = get_symbol_params(eff_symbol)
    macd_fast = int(sym_params.get("fast", settings.macd_fast))
    macd_slow = int(sym_params.get("slow", settings.macd_slow))
    macd_signal = int(sym_params.get("signal", settings.macd_signal))
    adx_len = int(sym_params.get("adx_len", settings.adx_len))

    df["macd"] = np.nan
    df["macd_signal"] = np.nan
    df["macd_hist"] = np.nan
    df["adx"] = np.nan

    macd_df = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_df is not None and not macd_df.empty:
        macd_cols = macd_df.columns.tolist()
        if len(macd_cols) >= 3:
            df["macd"] = macd_df[macd_cols[0]]
            df["macd_signal"] = macd_df[macd_cols[1]]
            df["macd_hist"] = macd_df[macd_cols[2]]

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=int(adx_len))
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if c.startswith("ADX")]
        if adx_cols:
            df["adx"] = adx_df[adx_cols[0]]
    return df


def volatility_target_size(df: pd.DataFrame, *, symbol: Optional[str] = None):
    eff_symbol = _resolve_symbol(symbol)
    sym_params = get_symbol_params(eff_symbol)
    lb = int(sym_params.get("vol_lb", settings.vol_lookback))
    tgt = float(sym_params.get("target_vol", settings.vol_target))
    returns = df["close"].pct_change()
    vol = returns.rolling(lb).std()
    # simple inverse-vol scaling toward a target
    scale = (tgt / (vol.replace(0, np.nan))).clip(
        lower=settings.min_size, upper=settings.max_size
    )
    return scale.fillna(0.0)


def last_signal(df: pd.DataFrame, *, symbol: Optional[str] = None):
    if df is None or df.empty or len(df) < 2:
        return "hold"

    row = df.iloc[-1]
    prev = df.iloc[-2]
    eff_symbol = _resolve_symbol(symbol)
    sym_params = get_symbol_params(eff_symbol)
    adx_thr = int(
        sym_params.get(
            "adx_th", sym_params.get("adx_threshold", settings.adx_threshold)
        )
    )
    adx_value = row.get("adx")
    if adx_value is None or pd.isna(adx_value):
        adx_value = row.get("ADX")
    if adx_value is None or pd.isna(adx_value):
        return "hold"
    adx_ok = float(adx_value) > adx_thr

    # MACD histogram cross as signal (pandas-ta column)
    def _get_hist(source_row: pd.Series) -> Optional[float]:
        value = source_row.get("MACD_hist")
        if value is None or pd.isna(value):
            value = source_row.get("macd_hist")
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    macd_hist = _get_hist(row)
    macd_hist_prev = _get_hist(prev)

    if adx_ok and macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > 0 and macd_hist_prev <= 0:
            return "buy"
        if macd_hist < 0 and macd_hist_prev >= 0:
            return "sell"
    return "hold"


def compute_realized_vol(df: pd.DataFrame, *, symbol: Optional[str] = None):
    eff_symbol = _resolve_symbol(symbol)
    sym_params = get_symbol_params(eff_symbol)
    lb = int(sym_params.get("vol_lb", settings.vol_lookback))
    returns = df["close"].pct_change()
    vol = returns.rolling(lb).std()
    if len(vol.dropna()) == 0:
        return 0.0
    return float(vol.iloc[-1])
