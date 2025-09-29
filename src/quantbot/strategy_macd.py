from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class StrategyParams:
    macd_fast: int = 8
    macd_slow: int = 24
    macd_signal: int = 9
    adx_length: int = 14
    adx_threshold: float = 20.0
    macd_cross_grace_bars: int = 3
    macd_thrust_bars: int = 2
    vol_lookback: int = 30
    target_daily_vol: float = 0.02
    min_fraction: float = 0.0
    max_fraction: float = 1.0
    cooldown_bars: int = 0
    entry_order_type: str = "market"
    exit_order_type: str = "market"
    bar_minutes: int = 60


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
    df["macd"] = np.nan
    df["macd_signal"] = np.nan
    df["macd_hist"] = np.nan
    df["adx"] = np.nan

    macd_df = ta.macd(
        df["close"],
        fast=int(params.macd_fast),
        slow=int(params.macd_slow),
        signal=int(params.macd_signal),
    )
    if macd_df is not None and not macd_df.empty:
        macd_cols = macd_df.columns.tolist()
        if len(macd_cols) >= 3:
            df["macd"] = macd_df[macd_cols[0]]
            df["macd_signal"] = macd_df[macd_cols[1]]
            df["macd_hist"] = macd_df[macd_cols[2]]

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=int(params.adx_length))
    if adx_df is not None and not adx_df.empty:
        adx_cols = [c for c in adx_df.columns if c.startswith("ADX")]
        if adx_cols:
            df["adx"] = adx_df[adx_cols[0]]
    returns = df["close"].pct_change()
    df["returns"] = returns
    bars_per_day = _bars_per_day(params.bar_minutes)
    lookback = max(1, int(params.vol_lookback))
    realized = returns.rolling(lookback).apply(
        lambda window: float(np.sqrt(np.nanmean(np.square(window)))), raw=True
    ) * np.sqrt(bars_per_day)
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


def _coerce_datetime(
    value: Optional[datetime | pd.Timestamp | str],
) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def apply_cooldown(
    result: SignalResult,
    params: StrategyParams,
    *,
    last_exit_ts: Optional[datetime],
    current_ts: datetime,
) -> SignalResult:
    cooldown = max(0, int(params.cooldown_bars))
    meta = dict(result.meta)
    meta.setdefault("cooldown_bars", cooldown)
    meta.setdefault("cooldown_ok", True)
    if cooldown <= 0 or result.signal != Signal.BUY:
        meta.setdefault("cooldown_active", False)
        meta["cooldown_ok"] = True
        return replace(result, meta=meta)
    prev_ts = _coerce_datetime(last_exit_ts)
    cur_ts = _coerce_datetime(current_ts)
    if prev_ts is None or cur_ts is None:
        meta.update({"cooldown_active": False, "cooldown_ok": True})
        return replace(result, meta=meta)
    bar_minutes = max(1, int(params.bar_minutes or 1))
    elapsed_minutes = max(0.0, (cur_ts - prev_ts).total_seconds() / 60.0)
    bars_elapsed = int(elapsed_minutes // bar_minutes)
    meta.update(
        {
            "cooldown_active": bars_elapsed < cooldown,
            "cooldown_bars": cooldown,
            "bars_since_exit": bars_elapsed,
            "cooldown_ok": bars_elapsed >= cooldown,
        }
    )
    if bars_elapsed < cooldown:
        return replace(
            result,
            signal=Signal.HOLD,
            reason="cooldown_active",
            meta=meta,
        )
    return replace(result, meta=meta)


def compute_signal(
    df: pd.DataFrame,
    params: StrategyParams,
    *,
    position_state: str = "FLAT",
) -> SignalResult:
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

    pos_state = (position_state or "FLAT").upper()
    if pos_state not in {"FLAT", "LONG", "SHORT"}:
        pos_state = "FLAT"

    hist_series = df["macd_hist"].astype(float)
    grace_bars = max(0, int(getattr(params, "macd_cross_grace_bars", 0)))
    thrust_bars = max(0, int(getattr(params, "macd_thrust_bars", 0)))

    cross_up_now = hist > 0 and hist_prev <= 0
    cross_down_now = hist < 0 and hist_prev >= 0

    cross_up_within_k = False
    cross_down_within_k = False
    if grace_bars > 0 and len(hist_series.dropna()) >= 2:
        recent = hist_series.iloc[-(grace_bars + 1) :]
        shifted = recent.shift(1)
        flips_up = (shifted <= 0) & (recent > 0)
        flips_down = (shifted >= 0) & (recent < 0)
        cross_up_within_k = bool(flips_up.iloc[-grace_bars:].any())
        cross_down_within_k = bool(flips_down.iloc[-grace_bars:].any())

    thrust_up = False
    thrust_down = False
    if thrust_bars > 0 and len(hist_series.dropna()) >= thrust_bars:
        diff_series = hist_series.diff().fillna(0.0)
        tail = diff_series.iloc[-thrust_bars:]
        thrust_up = bool(hist > 0 and (tail > 0).all())
        thrust_down = bool(hist < 0 and (tail < 0).all())

    trend_ok = adx_ok
    want_long = trend_ok and (
        cross_up_now or cross_up_within_k or (pos_state == "FLAT" and thrust_up)
    )
    want_short = trend_ok and (
        cross_down_now or cross_down_within_k or (pos_state == "FLAT" and thrust_down)
    )

    signal = Signal.HOLD
    reason = "no_cross"
    if not adx_ok:
        reason = "adx_below_threshold"
    elif not vol_ok:
        reason = "vol_scale_zero"
    else:
        if pos_state in {"FLAT", "SHORT"} and want_long:
            signal = Signal.BUY
            if cross_up_now:
                reason = "macd_cross_up"
            elif cross_up_within_k:
                reason = "macd_cross_recent"
            elif thrust_up:
                reason = "macd_thrust_up"
            else:
                reason = "trend_follow_buy"
        elif pos_state in {"FLAT", "LONG"} and want_short:
            signal = Signal.SELL
            if cross_down_now:
                reason = "macd_cross_down"
            elif cross_down_within_k:
                reason = "macd_cross_recent"
            elif thrust_down:
                reason = "macd_thrust_down"
            else:
                reason = "trend_follow_sell"
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
        meta={
            "adx_ok": adx_ok,
            "vol_ok": vol_ok,
            "trend_ok": trend_ok,
            "cross_up": cross_up_now or cross_up_within_k,
            "cross_down": cross_down_now or cross_down_within_k,
            "cross_up_now": cross_up_now,
            "cross_down_now": cross_down_now,
            "cross_up_within_k": cross_up_within_k,
            "cross_down_within_k": cross_down_within_k,
            "thrust_up": thrust_up,
            "thrust_down": thrust_down,
            "position_state": pos_state,
            "want_long": want_long,
            "want_short": want_short,
            "cooldown_ok": True,
            "macd_cross_grace_bars": grace_bars,
            "macd_thrust_bars": thrust_bars,
            "position": pos_state,
        },
    )


def compute_signal_history(
    df: pd.DataFrame,
    params: StrategyParams,
    *,
    lookback: int = 10,
) -> List[SignalResult]:
    results: List[SignalResult] = []
    last_exit_ts: Optional[datetime] = None
    position_state = "FLAT"
    for idx in range(len(df)):
        window = df.iloc[: idx + 1]
        if len(window) < 2:
            continue
        res = compute_signal(window, params, position_state=position_state)
        ts = window.index[-1]
        current_ts = (
            ts.to_pydatetime()
            if hasattr(ts, "to_pydatetime")
            else _coerce_datetime(ts) or datetime.fromisoformat(str(ts))
        )
        res = apply_cooldown(
            res, params, last_exit_ts=last_exit_ts, current_ts=current_ts
        )
        results.append(res)
        if res.signal == Signal.BUY:
            position_state = "LONG"
        elif res.signal == Signal.SELL:
            position_state = "SHORT"
        if res.signal == Signal.SELL:
            last_exit_ts = current_ts
    return results[-lookback:]


__all__ = [
    "StrategyParams",
    "Signal",
    "SignalResult",
    "compute_indicators",
    "compute_signal",
    "compute_signal_history",
    "position_sizer",
    "apply_cooldown",
]
