from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import talib  # type: ignore


@dataclass
class StrategyParams:
    macd_fast: int = 8
    macd_slow: int = 24
    macd_signal: int = 9
    adx_length: int = 14
    adx_threshold: float = 20.0
    vol_lookback: int = 20
    target_daily_vol: float = 0.02
    min_fraction: float = 0.0
    max_fraction: float = 1.0
    bar_minutes: int = 30


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    signal: Signal
    reason: str
    volatility: float
    volatility_scale: float
    macd_hist: float
    macd_hist_prev: float
    adx: float
    meta: Dict[str, object]


OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _to_dataframe(data: Sequence[Sequence[float]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(list(data), columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        df = df.drop(columns=["timestamp"])
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.astype(float)
    df = df.sort_index()
    return df


def _bars_per_day(bar_minutes: int) -> int:
    if bar_minutes <= 0:
        return 48
    return max(1, int(round(1440 / bar_minutes)))


def compute_indicators(
    data: Sequence[Sequence[float]] | pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    df = _to_dataframe(data)
    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    macd, macd_signal, macd_hist = talib.MACD(
        closes,
        fastperiod=int(params.macd_fast),
        slowperiod=int(params.macd_slow),
        signalperiod=int(params.macd_signal),
    )
    adx = talib.ADX(
        highs,
        lows,
        closes,
        timeperiod=int(params.adx_length),
    )
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["adx"] = adx
    returns = df["close"].pct_change()
    df["returns"] = returns
    bars_per_day = _bars_per_day(params.bar_minutes)
    realized = returns.rolling(int(params.vol_lookback)).std() * np.sqrt(bars_per_day)
    df["realized_vol"] = realized
    return df


def position_sizer(
    target_vol: float,
    vol_estimate: float,
    *,
    min_fraction: float,
    max_fraction: float,
) -> float:
    if vol_estimate is None or np.isnan(vol_estimate) or vol_estimate <= 0:
        return 0.0
    return float(np.clip(target_vol / vol_estimate, min_fraction, max_fraction))


def compute_signal(df: pd.DataFrame, params: StrategyParams) -> SignalResult:
    if df.empty or len(df) < 2:
        return SignalResult(
            signal=Signal.HOLD,
            reason="insufficient_data",
            volatility=0.0,
            volatility_scale=0.0,
            macd_hist=0.0,
            macd_hist_prev=0.0,
            adx=0.0,
            meta={"adx_ok": False, "vol_ok": False},
        )
    row = df.iloc[-1]
    prev = df.iloc[-2]
    hist = float(row.get("macd_hist", np.nan))
    hist_prev = float(prev.get("macd_hist", np.nan))
    adx = float(row.get("adx", np.nan))
    realized_vol = float(row.get("realized_vol", np.nan))

    adx_ok = not np.isnan(adx) and adx >= params.adx_threshold
    vol_scale = position_sizer(
        params.target_daily_vol,
        realized_vol,
        min_fraction=params.min_fraction,
        max_fraction=params.max_fraction,
    )
    vol_ok = vol_scale > 0

    cross_up = hist > 0 and hist_prev <= 0
    cross_down = hist < 0 and hist_prev >= 0

    signal = Signal.HOLD
    reason = "no_cross"
    if not adx_ok:
        reason = "adx_below_threshold"
    elif not vol_ok:
        reason = "vol_scale_zero"
    else:
        if cross_up:
            signal = Signal.BUY
            reason = "macd_cross_up"
        elif cross_down:
            signal = Signal.SELL
            reason = "macd_cross_down"
        else:
            reason = "hold"

    return SignalResult(
        signal=signal,
        reason=reason,
        volatility=realized_vol if not np.isnan(realized_vol) else 0.0,
        volatility_scale=vol_scale,
        macd_hist=hist if not np.isnan(hist) else 0.0,
        macd_hist_prev=hist_prev if not np.isnan(hist_prev) else 0.0,
        adx=adx if not np.isnan(adx) else 0.0,
        meta={"adx_ok": adx_ok, "vol_ok": vol_ok, "cross_up": cross_up, "cross_down": cross_down},
    )


def compute_signal_history(
    df: pd.DataFrame,
    params: StrategyParams,
    *,
    lookback: int = 10,
) -> List[SignalResult]:
    results: List[SignalResult] = []
    for idx in range(len(df)):
        window = df.iloc[: idx + 1]
        if len(window) < 2:
            continue
        results.append(compute_signal(window, params))
    return results[-lookback:]


__all__ = [
    "StrategyParams",
    "Signal",
    "SignalResult",
    "compute_indicators",
    "compute_signal",
    "compute_signal_history",
    "position_sizer",
]
