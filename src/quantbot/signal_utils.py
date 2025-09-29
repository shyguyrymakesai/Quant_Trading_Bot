"""Utilities for safely consuming indicator data windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowStatus:
    """Result of validating a historical data window."""

    ok: bool
    reason: str
    bars: int
    needed: int
    have_closed: bool


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def bars_needed_contract(
    *,
    macd_fast: object,
    macd_slow: object,
    macd_signal: object,
    adx_length: object,
    vol_lookback: object,
    macd_cross_grace_bars: object = 0,
    macd_thrust_bars: object = 0,
    minimum: int = 2,
) -> int:
    """Return a conservative bar count required for stable indicators."""

    slow = _coerce_int(macd_slow)
    signal = _coerce_int(macd_signal)
    adx_len = _coerce_int(adx_length)
    vol_lb = _coerce_int(vol_lookback)
    grace = _coerce_int(macd_cross_grace_bars)
    thrust = _coerce_int(macd_thrust_bars)

    fast = _coerce_int(macd_fast)
    macd_requirement = max(slow, fast) + max(signal, 1)
    adx_requirement = max(adx_len * 2, adx_len + 1)
    vol_requirement = max(vol_lb + 1, 2)
    cross_requirement = grace + 2
    thrust_requirement = thrust + 2

    return max(
        minimum,
        macd_requirement,
        adx_requirement,
        vol_requirement,
        cross_requirement,
        thrust_requirement,
    )


def _to_timestamp(value: pd.Timestamp | datetime) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return pd.Timestamp(value)
        return pd.Timestamp(value, tz="UTC")
    return pd.Timestamp(value)


def validate_window(
    df: pd.DataFrame,
    *,
    min_bars: int,
    as_of: Optional[datetime] = None,
    bar_minutes: Optional[int] = None,
    required_columns: Optional[Sequence[str]] = None,
) -> WindowStatus:
    """Validate that a dataframe has enough closed, indicator-ready bars."""

    if df is None or df.empty:
        return WindowStatus(False, "no_data", 0, min_bars, False)

    bars = int(len(df))
    have_closed = True
    if as_of is not None and bar_minutes:
        last_ts = _to_timestamp(df.index[-1])
        ref = _to_timestamp(as_of)
        delta_minutes = (ref - last_ts).total_seconds() / 60.0
        have_closed = delta_minutes >= float(bar_minutes)

    if bars < min_bars:
        return WindowStatus(False, "insufficient_bars", bars, min_bars, have_closed)

    if not have_closed:
        return WindowStatus(False, "partial_bar", bars, min_bars, have_closed)

    if required_columns:
        required = [c for c in required_columns if c in df.columns]
        if required:
            ready = df.dropna(subset=required)
            if len(ready) < 2:
                return WindowStatus(False, "indicators_not_ready", bars, min_bars, have_closed)
            latest_ready = ready.index[-1]
            if latest_ready != df.index[-1]:
                return WindowStatus(
                    False,
                    "indicators_not_ready",
                    bars,
                    min_bars,
                    have_closed,
                )

    return WindowStatus(True, "ok", bars, min_bars, have_closed)


def safe_tail(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return the last ``count`` rows without raising for short frames."""

    if df is None or df.empty or count <= 0:
        return df.iloc[0:0]
    count = int(count)
    if count >= len(df):
        return df.copy()
    return df.iloc[-count:].copy()


def safe_float(value: object, *, default: float = np.nan) -> float:
    """Convert a value to float while keeping NaN semantics."""

    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(result) and not np.isnan(default):
        return float(default)
    return result


__all__ = [
    "WindowStatus",
    "bars_needed_contract",
    "validate_window",
    "safe_tail",
    "safe_float",
]
