from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.strategy_macd import (
    Signal,
    StrategyParams,
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
    result = compute_signal(df, params)
    assert result.signal == Signal.BUY
    assert result.meta["adx_ok"] is True
    assert result.meta["vol_ok"] is True


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
    result_down = compute_signal(df_down, params)
    assert result_down.signal == Signal.SELL

    df_low_adx = df_down.copy()
    df_low_adx.loc[:, "adx"] = [10, 10]
    result_low = compute_signal(df_low_adx, params)
    assert result_low.signal == Signal.HOLD
    assert result_low.reason == "adx_below_threshold"
