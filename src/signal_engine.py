import pandas as pd
import pandas_ta as ta
import numpy as np
from .config import settings

def compute_indicators(ohlcv: list):
    """ohlcv: list of [ts, open, high, low, close, volume]"""
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    macd = ta.macd(df['close'], fast=settings.macd_fast, slow=settings.macd_slow, signal=settings.macd_signal)
    adx = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=settings.adx_len)
    df = pd.concat([df, macd, adx], axis=1)
    return df

def volatility_target_size(df: pd.DataFrame):
    returns = df['close'].pct_change()
    vol = returns.rolling(settings.vol_lookback).std()
    # simple inverse-vol scaling toward a target
    scale = (settings.vol_target / (vol.replace(0, np.nan))).clip(lower= settings.min_size, upper=settings.max_size)
    return scale.fillna(0.0)

def last_signal(df: pd.DataFrame):
    row = df.iloc[-1]
    prev = df.iloc[-2]
    adx_ok = row.get('ADX_14', row.get('ADX_'+str(settings.adx_len), 0)) > settings.adx_threshold
    # MACD histogram cross as signal
    macd_hist = row.get('MACDh_'+str(settings.macd_fast)+'_'+str(settings.macd_slow)+'_'+str(settings.macd_signal))
    macd_hist_prev = prev.get('MACDh_'+str(settings.macd_fast)+'_'+str(settings.macd_slow)+'_'+str(settings.macd_signal))
    if adx_ok and macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > 0 and macd_hist_prev <= 0:
            return "buy"
        if macd_hist < 0 and macd_hist_prev >= 0:
            return "sell"
    return "hold"
