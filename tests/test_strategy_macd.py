from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import pytest
from datetime import datetime, timedelta, timezone

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from quantbot.strategy_macd import (
    Signal,
    SignalResult,
    StrategyParams,
    apply_cooldown,
    compute_indicators,
    compute_signal,
    position_sizer,
)


def make_dummy_ohlcv(periods: int = 60):
    index = pd.date_range("2024-01-01", periods=periods, freq="30min", tz="UTC")
    data = []
    price = 100.0
    for ts in index:
        high = price + 1.0
        low = price - 1.0
        close = price
        data.append([int(ts.timestamp() * 1000), price, high, low, close, 10.0])
        price += 0.5
    return data


def test_compute_indicators_columns():
    params = StrategyParams()
    df = compute_indicators(make_dummy_ohlcv(), params)
    assert {"macd_hist", "adx", "realized_vol"}.issubset(df.columns)
    assert df.index.tz is not None


def test_position_sizer_scaling():
    scale = position_sizer(0.02, 0.05, min_fraction=0.0, max_fraction=1.0)
    assert pytest.approx(scale, rel=1e-3) == 0.4


def test_compute_signal_buy():
    params = StrategyParams(adx_threshold=20, target_daily_vol=0.02)
    idx = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "macd_hist": [-0.1, 0.2],
            "adx": [25, 25],
            "realized_vol": [0.01, 0.01],
            "close": [100.0, 101.0],
        },
        index=idx,
    )
    result = compute_signal(df, params, position_state="FLAT")
    assert result.signal == Signal.BUY
    assert result.meta["adx_ok"] is True
    assert result.meta["vol_ok"] is True
    assert result.meta["cross_up_now"] is True
    assert result.reason == "macd_cross_up"


def test_compute_signal_sell_and_gating():
    params = StrategyParams(adx_threshold=25, target_daily_vol=0.02)
    idx = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    df_down = pd.DataFrame(
        {
            "macd_hist": [0.3, -0.2],
            "adx": [30, 30],
            "realized_vol": [0.01, 0.01],
            "close": [101.0, 100.0],
        },
        index=idx,
    )
    result_down = compute_signal(df_down, params, position_state="LONG")
    assert result_down.signal == Signal.SELL
    assert result_down.reason in {"macd_cross_down", "macd_cross_recent"}

    df_low_adx = df_down.copy()
    df_low_adx.loc[:, "adx"] = [10, 10]
    result_low = compute_signal(df_low_adx, params, position_state="LONG")
    assert result_low.signal == Signal.HOLD
    assert result_low.reason == "adx_below_threshold"


def test_grace_window_triggers_buy_when_cross_recent():
    params = StrategyParams(
        adx_threshold=20,
        target_daily_vol=0.02,
        macd_cross_grace_bars=3,
        macd_thrust_bars=0,
    )
    idx = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "macd_hist": [-0.2, 0.1, 0.2],
            "adx": [30, 30, 30],
            "realized_vol": [0.01, 0.01, 0.01],
            "close": [100.0, 101.0, 102.0],
        },
        index=idx,
    )
    result = compute_signal(df, params, position_state="FLAT")
    assert result.signal == Signal.BUY
    assert result.reason == "macd_cross_recent"
    assert result.meta["cross_up_within_k"] is True


def test_thrust_entry_when_hist_rising():
    params = StrategyParams(
        adx_threshold=20,
        target_daily_vol=0.02,
        macd_cross_grace_bars=0,
        macd_thrust_bars=2,
    )
    idx = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "macd_hist": [0.05, 0.1, 0.2],
            "adx": [35, 35, 35],
            "realized_vol": [0.01, 0.01, 0.01],
            "close": [100.0, 100.5, 101.0],
        },
        index=idx,
    )
    result = compute_signal(df, params, position_state="FLAT")
    assert result.signal == Signal.BUY
    assert result.reason == "macd_thrust_up"
    assert result.meta["thrust_up"] is True


def make_noisy_ohlcv(periods: int = 60):
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    data = []
    base = 100.0
    for i, ts in enumerate(index):
        swing = 5.0 * np.sin(i / 3.0)
        price = base + swing
        high = price + 1.5
        low = price - 1.5
        close = price
        data.append([int(ts.timestamp() * 1000), price, high, low, close, 15.0])
        base += 0.2 * (-1) ** i
    return data



def test_apply_cooldown_blocks_buy_within_window():
    params = StrategyParams(cooldown_bars=2, bar_minutes=60)
    base_result = SignalResult(
        signal=Signal.BUY,
        reason="macd_cross_up",
        volatility=0.01,
        volatility_scale=1.0,
        macd_hist=0.2,
        macd_hist_prev=-0.1,
        adx=25.0,
        meta={},
    )
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    last_exit = now - timedelta(hours=1)
    gated = apply_cooldown(base_result, params, last_exit_ts=last_exit, current_ts=now)
    assert gated.signal == Signal.HOLD
    assert gated.reason == "cooldown_active"
    assert gated.meta["cooldown_active"] is True
    assert gated.meta["cooldown_ok"] is False
    assert gated.meta["bars_since_exit"] == 1


def test_apply_cooldown_allows_entry_after_window():
    params = StrategyParams(cooldown_bars=2, bar_minutes=60)
    base_result = SignalResult(
        signal=Signal.BUY,
        reason="macd_cross_up",
        volatility=0.01,
        volatility_scale=1.0,
        macd_hist=0.2,
        macd_hist_prev=-0.1,
        adx=25.0,
        meta={},
    )
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    last_exit = now - timedelta(hours=4)
    allowed = apply_cooldown(base_result, params, last_exit_ts=last_exit, current_ts=now)
    assert allowed.signal == Signal.BUY
    assert allowed.meta["cooldown_active"] is False
    assert allowed.meta["cooldown_ok"] is True


def test_compute_indicators_respects_vol_lookback():
    noisy = make_noisy_ohlcv(90)
    params_short = StrategyParams(vol_lookback=5, bar_minutes=60)
    params_long = StrategyParams(vol_lookback=30, bar_minutes=60)
    df_short = compute_indicators(noisy, params_short)
    df_long = compute_indicators(noisy, params_long)
    assert df_long["realized_vol"].iloc[-1] <= df_short["realized_vol"].iloc[-1]

