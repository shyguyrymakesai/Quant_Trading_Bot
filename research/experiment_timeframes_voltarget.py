"""
Experiment: 30m vs 1h vs 4h with daily-scaled volatility targeting and WFV.

Design highlights
- Symbols: default BTCUSDT (optionally ETHUSDT)
- Period: default 2018-01-01 to 2025-09-01 (intersection across TFs)
- Costs: round-trip fees+slippage in bps (default 8 bps total, split per side)
- Rules: MACD cross/exit, ADX filter, volatility targeting to a daily target
- WF: train 12m, test 3m, roll 3m; pick best on IS by Sharpe, lock for OOS

Outputs
- Aggregated per-TF WFV summary CSV at research/reports/wfv_summary.csv
- Optional per-segment details printed to console

Notes & caveats
- Free data limitations: 30m interval history is limited on common sources (e.g.,
  yfinance). We compute the date-span intersection of all TF datasets. If 30m is
  short, it will constrain the span for all TFs. You can swap in your own data
  provider if you have a long 30m history.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Data sources
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

# TA and Backtesting
try:
    import talib  # type: ignore
except Exception as e:  # pragma: no cover
    talib = None

from backtesting import Backtest, Strategy  # type: ignore


# ================================
# Constants and configuration
# ================================
TIMEFRAMES = ["30m", "1h", "4h"]
BARS_PER_DAY = {"30m": 48, "1h": 24, "4h": 6}

# A compact parameter grid to keep runtime reasonable by default
GRID = {
    "fast": [8, 12],
    "slow": [24, 26, 35],
    "signal": [6, 9],
    "adx_len": [14],
    "adx_th": [22, 25, 30, 35],
    "vol_lb": [20, 30],
    "daily_vol_target": [0.015, 0.02],
    "risk_frac": [0.1, 0.2],
    "cooldown_bars": [0, 4, 6],
}

DEFAULT_FEES_BPS = 8.0  # round-trip bps
DEFAULT_INITIAL_CASH = 10_000_000.0


# ================================
# Utilities
# ================================


def _to_ccxt_symbol(sym: str) -> str:
    sym = sym.upper().replace("-", "").replace("/", "")
    if sym.endswith("USDT"):
        base = sym[:-4]
        return f"{base}/USDT"
    if sym.endswith("USD"):
        base = sym[:-3]
        return f"{base}/USD"
    if len(sym) > 3:
        return f"{sym[:-3]}/{sym[-3:]}"
    return sym


def _to_yf_symbol(sym: str) -> str:
    sym = sym.upper()
    if sym.endswith("USDT"):
        base = sym[:-4]
        # yfinance typically has USD pairs
        return f"{base}-USD"
    if sym.endswith("USD"):
        base = sym[:-3]
        return f"{base}-USD"
    return f"{sym}-USD"


def _pandas_resample_rule(tf: str) -> str:
    return {"30m": "30T", "1h": "1H", "4h": "4H"}[tf]


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # Assumes df has columns: open, high, low, close, volume and DatetimeIndex UTC
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    out = out.dropna()
    return out


def _fetch_ccxt_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    max_bars: int = 200000,
) -> Optional[pd.DataFrame]:
    if ccxt is None:
        return None
    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    except Exception:
        return None

    if not hasattr(ex, "fetch_ohlcv"):
        return None

    # Ensure market exists
    try:
        ex.load_markets()
    except Exception:
        pass

    # Adjust symbol for coinbase which often uses USD rather than USDT
    sym_try = [symbol]
    if exchange_id.startswith("coinbase") and symbol.endswith("/USDT"):
        sym_try.append(symbol.replace("/USDT", "/USD"))

    data: List[List[float]] = []
    for sym in sym_try:
        try:
            since = since_ms
            while True:
                batch = ex.fetch_ohlcv(
                    sym, timeframe=timeframe, since=since, limit=1000
                )
                if not batch:
                    break
                data.extend(batch)
                if until_ms is not None and batch[-1][0] >= until_ms:
                    break
                since = batch[-1][0] + 1
                if len(data) >= max_bars:
                    break
            if data:
                break
        except Exception:
            data = []
            continue

    if not data:
        return None

    arr = np.array(data)
    df = pd.DataFrame(arr, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _fetch_yf_ohlcv(
    symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        # yfinance intraday intervals have history limits (approx 730 days for 1h/4h).
        # Fetch in chunks to cover longer ranges.
        intraday = interval in {"30m", "1h", "4h"}
        if intraday:
            step_days = 700
            parts = []
            cur = start
            while cur < end:
                nxt = min(end, cur + pd.Timedelta(days=step_days))
                part = yf.download(
                    symbol,
                    start=cur.tz_convert(None),
                    end=nxt.tz_convert(None),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                )
                if part is not None and not part.empty:
                    parts.append(part)
                cur = nxt
            if not parts:
                return None
            df = pd.concat(parts).sort_index()
            df = df[~df.index.duplicated(keep="first")]
        else:
            df = yf.download(
                symbol,
                start=start.tz_convert(None),
                end=end.tz_convert(None),
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        if df is None or df.empty:
            return None
        # yfinance columns: Open High Low Close Adj Close Volume
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception:
        return None


def _fetch_all_tfs_from_ccxt(
    symbol_in: str, start: pd.Timestamp, end: pd.Timestamp, ex_id: str
) -> Optional[Dict[str, pd.DataFrame]]:
    symbol_ccxt = _to_ccxt_symbol(symbol_in)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    out: Dict[str, pd.DataFrame] = {}
    # Fetch in an order that helps resampling fallbacks
    for tf in ["1h", "4h", "30m"]:
        df = _fetch_ccxt_ohlcv(
            ex_id, symbol_ccxt, tf, start_ms, end_ms, max_bars=200000
        )
        if df is None or len(df) <= 50:
            # Try resample fallbacks
            if tf == "4h":
                # Try 1h then resample to 4h
                base = out.get("1h")
                if base is None:
                    base = _fetch_ccxt_ohlcv(
                        ex_id, symbol_ccxt, "1h", start_ms, end_ms, max_bars=200000
                    )
                if base is not None and len(base) > 50:
                    df = _resample_ohlcv(base, "4H")
            elif tf == "30m":
                # Try 15m then resample to 30m
                base15 = _fetch_ccxt_ohlcv(
                    ex_id, symbol_ccxt, "15m", start_ms, end_ms, max_bars=200000
                )
                if base15 is not None and len(base15) > 50:
                    df = _resample_ohlcv(base15, "30T")
            # If still None or too short, give up for this exchange
            if df is None or len(df) <= 50:
                return None
        out[tf] = df
    # Log and return
    for tf, df in out.items():
        print(f"CCXT source={ex_id} symbol={symbol_ccxt} tf={tf} bars={len(df)}")
    return out


def load_data_tf(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    """Load a single timeframe without cross-TF intersection.
    Prefer CCXT with optional resampling fallback; then yfinance.
    Returns a DataFrame with columns [open, high, low, close, volume].
    """
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)

    symbol_ccxt = _to_ccxt_symbol(symbol)
    start_ms = int(s.timestamp() * 1000)
    end_ms = int(e.timestamp() * 1000)

    # Try CCXT across a few exchanges, with resample fallbacks for 4h and 30m
    for ex_id in ["binance", "coinbase", "coinbaseadvanced"]:
        df = _fetch_ccxt_ohlcv(
            ex_id, symbol_ccxt, tf, start_ms, end_ms, max_bars=200000
        )
        if (df is None or len(df) <= 50) and tf == "4h":
            # try 1h then resample to 4h
            base = _fetch_ccxt_ohlcv(
                ex_id, symbol_ccxt, "1h", start_ms, end_ms, max_bars=200000
            )
            if base is not None and len(base) > 50:
                df = _resample_ohlcv(base, "4H")
        if (df is None or len(df) <= 50) and tf == "30m":
            # try 15m then resample to 30m
            base15 = _fetch_ccxt_ohlcv(
                ex_id, symbol_ccxt, "15m", start_ms, end_ms, max_bars=200000
            )
            if base15 is not None and len(base15) > 50:
                df = _resample_ohlcv(base15, "30T")

        if df is not None and len(df) > 50:
            return df

    # Fallback: yfinance exact interval
    yf_sym = _to_yf_symbol(symbol)
    yf_interval = {"30m": "30m", "1h": "1h", "4h": "4h"}[tf]
    df_yf = _fetch_yf_ohlcv(yf_sym, yf_interval, s, e)
    if df_yf is None or df_yf.empty:
        raise RuntimeError(f"No data for {symbol} {tf} in [{start},{end}]")
    return df_yf


def load_data_all_tf(symbol: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)

    # Try unified CCXT source first to guarantee overlap across TFs
    for ex_id in ["binance", "coinbase", "coinbaseadvanced"]:
        dfs = _fetch_all_tfs_from_ccxt(symbol, s, e, ex_id)
        if dfs is not None:
            # Intersect spans across TFs
            latest_start = max(v.index.min() for v in dfs.values())
            earliest_end = min(v.index.max() for v in dfs.values())
            if not (
                pd.isna(latest_start)
                or pd.isna(earliest_end)
                or latest_start >= earliest_end
            ):
                out: Dict[str, pd.DataFrame] = {}
                for tf, df in dfs.items():
                    df2 = df.loc[
                        (df.index >= latest_start) & (df.index <= earliest_end)
                    ].copy()
                    df2 = df2[~df2.index.duplicated(keep="first")]
                    out[tf] = df2
                return out

    # Fallback: yfinance for all TFs
    dfs_yf: Dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        yf_sym = _to_yf_symbol(symbol)
        yf_interval = {"30m": "30m", "1h": "1h", "4h": "4h"}[tf]
        df = _fetch_yf_ohlcv(yf_sym, yf_interval, s, e)
        if df is None or df.empty:
            continue
        print(f"YF source symbol={yf_sym} tf={tf} bars={len(df)}")
        dfs_yf[tf] = df

    if not dfs_yf:
        raise RuntimeError(
            f"No data fetched for {symbol} across timeframes {TIMEFRAMES}"
        )

    latest_start = max(v.index.min() for v in dfs_yf.values())
    earliest_end = min(v.index.max() for v in dfs_yf.values())
    if pd.isna(latest_start) or pd.isna(earliest_end) or latest_start >= earliest_end:
        raise RuntimeError("Insufficient intersection across timeframes (yfinance)")

    out2: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs_yf.items():
        df2 = df.loc[(df.index >= latest_start) & (df.index <= earliest_end)].copy()
        df2 = df2[~df2.index.duplicated(keep="first")]
        out2[tf] = df2
    return out2


# ================================
# Strategy
# ================================
class MACD_ADX_VolTarget(Strategy):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    adx_len: int = 14
    adx_th: float = 20.0
    risk_frac: float = 0.5
    daily_vol_target: float = 0.02
    vol_lb: int = 30
    cooldown_bars: int = 0
    _last_exit_bar: int = -10**9

    def init(self):  # type: ignore
        if talib is None:
            raise ImportError("TA-Lib is required for this strategy")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        macd, macdsig, _ = self.I(
            lambda c, f, s, sig: talib.MACD(
                c, fastperiod=f, slowperiod=s, signalperiod=sig
            ),
            close,
            self.fast,
            self.slow,
            self.signal,
        )
        self.macd = macd
        self.macdsig = macdsig
        self.hist = self.I(lambda a, b: a - b, self.macd, self.macdsig)

        adx = self.I(
            lambda h, l, c, n: talib.ADX(h, l, c, timeperiod=n),
            high,
            low,
            close,
            self.adx_len,
        )
        self.adx = adx
        self._last_exit_bar = -10**9

        # Vol estimates (rolling std of returns)
        ret = self.I(lambda c: pd.Series(c).pct_change().fillna(0.0).values, close)
        # Rolling std with configurable lookback as bar vol proxy
        self.bar_vol = self.I(
            lambda r, n: pd.Series(r).rolling(int(n)).std().fillna(0.0).values,
            ret,
            self.vol_lb,
        )

    def _calc_units(self) -> int:
        price = float(self.data.Close[-1])
        if not np.isfinite(price) or price <= 0:
            return 0
        # Bars per day based on timeframe length derived from index
        # We infer from approximate average spacing of last 50 bars
        idx = self.data.index
        if len(idx) >= 2:
            deltas = pd.Series(idx[-50:]).diff().dropna()
            avg_minutes = (
                deltas.dt.total_seconds().mean() / 60.0 if len(deltas) else 60.0
            )
        else:
            avg_minutes = 60.0
        bars_per_day = max(1.0, 1440.0 / avg_minutes)

        bar_vol = float(self.bar_vol[-1]) if np.isfinite(self.bar_vol[-1]) else 0.0
        daily_vol_est = bar_vol * math.sqrt(bars_per_day)
        if daily_vol_est <= 0:
            scale = 0.0
        else:
            scale = float(self.daily_vol_target) / daily_vol_est
        scale = float(np.clip(scale, 0.1, 10.0))

        equity = float(self.equity)
        base_units = (equity * float(self.risk_frac)) / price
        units = int(max(0, math.floor(base_units * scale)))
        return units

    def next(self):  # type: ignore
        min_bars = max(int(self.slow), int(self.adx_len), int(self.vol_lb))
        if len(self.data) < min_bars + 5:
            return
        idx_pos = len(self.data) - 1
        hist_prev = float(self.hist[-2])
        hist_now = float(self.hist[-1])
        adx_now = float(self.adx[-1])

        bullish_cross = (
            hist_prev <= 0 and hist_now > 0 and adx_now >= float(self.adx_th)
        )
        bearish_cross = hist_prev >= 0 and hist_now < 0

        cooldown = max(0, int(self.cooldown_bars))
        if not self.position:
            if idx_pos - self._last_exit_bar < cooldown:
                return
            if bullish_cross:
                size = self._calc_units()
                if size > 0:
                    self.buy(size=size)
        else:
            if bearish_cross:
                self.position.close()
                self._last_exit_bar = idx_pos


# ================================
# Metrics and backtest glue
# ================================


def sharpe_annual_from_bar_returns(ret: pd.Series, bars_per_day: float) -> float:
    ret = pd.Series(ret, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2:
        return float("nan")
    mu = ret.mean()
    sd = ret.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    bars_per_year = bars_per_day * 365.0
    return float((mu / sd) * math.sqrt(bars_per_year))


def _extract_equity_and_trades(stats) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    # backtesting.Backtest.run returns a Stats object (dict-like)
    eq = getattr(stats, "_equity_curve", None)
    if eq is not None and isinstance(eq, pd.DataFrame) and "Equity" in eq.columns:
        equity = eq["Equity"].copy()
    else:
        # fallback try
        equity = pd.Series(dtype=float)
    trades = getattr(stats, "_trades", None)
    return equity, trades


def _compute_turnover_and_fees(
    trades: Optional[pd.DataFrame], commission: float, initial_cash: float
) -> Tuple[float, float]:
    if trades is None or trades.empty:
        return 0.0, 0.0
    # Approximate: turnover counts both entry and exit notionals
    entry_notional = (trades["EntryPrice"].abs() * trades["Size"].abs()).sum()
    exit_notional = (trades["ExitPrice"].abs() * trades["Size"].abs()).sum()
    turnover = float((entry_notional + exit_notional) / max(1e-12, initial_cash))
    # Fees approx = commission per side * notional per side
    fees_paid = float(commission * (entry_notional + exit_notional))
    return turnover, fees_paid


def backtest_once(
    df: pd.DataFrame, params: Dict, commission: float, initial_cash: float
) -> Dict:
    bt = Backtest(
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        ),
        MACD_ADX_VolTarget,
        cash=initial_cash,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run(
        fast=params["fast"],
        slow=params["slow"],
        signal=params["signal"],
        adx_len=params["adx_len"],
        adx_th=params["adx_th"],
        risk_frac=params["risk_frac"],
        daily_vol_target=params["daily_vol_target"],
        vol_lb=params.get("vol_lb", MACD_ADX_VolTarget.vol_lb),
        cooldown_bars=params.get("cooldown_bars", MACD_ADX_VolTarget.cooldown_bars),
    )
    equity, trades = _extract_equity_and_trades(stats)
    eq = equity.astype(float)
    ret_bar = eq.pct_change().fillna(0.0)

    # Pull some metrics
    maxdd_pct = float(stats.get("Max. Drawdown [%]", np.nan))
    exposure_pct = float(stats.get("Exposure Time [%]", np.nan))
    trades_count = int(stats.get("# Trades", 0))
    hit_rate = float(stats.get("Win Rate [%]", np.nan))

    # Average holding time in hours if available
    avg_hold_hours = np.nan
    if trades is not None and not trades.empty and "Duration" in trades.columns:
        try:
            # Duration is Timedelta
            avg_hold_hours = float(
                pd.to_timedelta(trades["Duration"]).dt.total_seconds().mean() / 3600.0
            )
        except Exception:
            avg_hold_hours = float("nan")

    turnover, fees_paid = _compute_turnover_and_fees(trades, commission, initial_cash)

    return {
        "ret_bar": ret_bar,
        "equity": eq,
        "maxdd_pct": maxdd_pct,
        "exposure_pct": exposure_pct,
        "trades": trades_count,
        "hit_rate": hit_rate,
        "avg_hold_hours": avg_hold_hours,
        "turnover": turnover,
        "fees_paid": fees_paid,
        "trades_df": trades,
    }


# ================================
# Walk-forward evaluation
# ================================
@dataclass
class Segment:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def walk_forward_splits(
    index: pd.DatetimeIndex,
    is_months: int = 12,
    oos_months: int = 3,
    roll_months: int = 3,
) -> List[Segment]:
    idx = (
        pd.DatetimeIndex(index).tz_convert("UTC")
        if index.tz is not None
        else pd.DatetimeIndex(index, tz="UTC")
    )
    start = idx.min().normalize()
    end = idx.max().normalize()

    segments: List[Segment] = []
    cur_is_start = start
    while True:
        is_end = cur_is_start + pd.DateOffset(months=is_months)
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > end:
            break
        segments.append(Segment(cur_is_start, is_end, oos_start, oos_end))
        cur_is_start = cur_is_start + pd.DateOffset(months=roll_months)
    return segments


def _iter_param_grid(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for tup in product(*vals):
        combos.append({k: v for k, v in zip(keys, tup)})
    return combos


def evaluate_segment(
    df: pd.DataFrame,
    timeframe: str,
    seg: Segment,
    grid: Dict[str, List],
    fees_bps: float,
    initial_cash: float,
) -> Dict:
    # Slice IS and OOS
    is_df = df.loc[(df.index >= seg.is_start) & (df.index < seg.is_end)]
    oos_df = df.loc[(df.index >= seg.oos_start) & (df.index < seg.oos_end)]

    if len(is_df) < 200 or len(oos_df) < 50:
        print(
            f"[SKIP] {timeframe} IS={len(is_df)} OOS={len(oos_df)} "
            f"seg=({seg.is_start}→{seg.is_end})/({seg.oos_start}→{seg.oos_end})",
            flush=True,
        )
        return {
            "oos_gross_sharpe": float("nan"),
            "oos_net_sharpe": float("nan"),
            "maxdd_pct": float("nan"),
            "trades": 0,
            "trades_per_month": float("nan"),
            "turnover": 0.0,
            "fees_pct_gross": float("nan"),
            "exposure": float("nan"),
            "hit_rate": float("nan"),
            "avg_hold": float("nan"),
        }

    bars_per_day = max(
        1.0,
        1440.0
        / max(
            1.0,
            (pd.Series(is_df.index).diff().dt.total_seconds().dropna().mean() or 3600.0)
            / 60.0,
        ),
    )

    # Select best on IS by net Sharpe (with commission half-sides)
    comm_per_side = float(fees_bps) / 2.0 / 10_000.0
    best_net_sharpe = -np.inf
    best_params = None

    for params in _iter_param_grid(grid):
        res_net = backtest_once(
            is_df, params, commission=comm_per_side, initial_cash=initial_cash
        )
        sharpe_net = sharpe_annual_from_bar_returns(res_net["ret_bar"], bars_per_day)
        if np.isfinite(sharpe_net) and sharpe_net > best_net_sharpe:
            best_net_sharpe = sharpe_net
            best_params = params

    if best_params is None:
        print(
            f"No viable params on IS segment for tf={timeframe} seg=[{seg.is_start} -> {seg.is_end}]",
            flush=True,
        )
        return {
            "oos_gross_sharpe": float("nan"),
            "oos_net_sharpe": float("nan"),
            "maxdd_pct": float("nan"),
            "trades": 0,
            "trades_per_month": float("nan"),
            "turnover": 0.0,
            "fees_pct_gross": float("nan"),
            "exposure": float("nan"),
            "hit_rate": float("nan"),
            "avg_hold": float("nan"),
        }

    # Evaluate on OOS: gross and net
    res_gross = backtest_once(
        oos_df, best_params, commission=0.0, initial_cash=initial_cash
    )
    res_net = backtest_once(
        oos_df, best_params, commission=comm_per_side, initial_cash=initial_cash
    )

    # bars_per_day recompute for oos_df to be accurate
    bars_per_day_oos = max(
        1.0,
        1440.0
        / max(
            1.0,
            (
                pd.Series(oos_df.index).diff().dt.total_seconds().dropna().mean()
                or 3600.0
            )
            / 60.0,
        ),
    )
    oos_gross_sharpe = sharpe_annual_from_bar_returns(
        res_gross["ret_bar"], bars_per_day_oos
    )
    oos_net_sharpe = sharpe_annual_from_bar_returns(
        res_net["ret_bar"], bars_per_day_oos
    )

    gross_pnl = (
        float(res_gross["equity"].iloc[-1] - res_gross["equity"].iloc[0])
        if len(res_gross["equity"]) > 1
        else 0.0
    )
    fees_paid = float(
        res_net["fees_paid"]
    )  # from turnover approximation with commission per side
    fees_pct_gross = float("nan")
    if abs(gross_pnl) > 1e-8:
        fees_pct_gross = float(np.clip(fees_paid / abs(gross_pnl), 0.0, 10.0))

    oos_days = max((seg.oos_end - seg.oos_start).days, 1)
    trades_per_month = float(res_net["trades"]) / (oos_days / 30.0) if oos_days > 0 else float("nan")

    return {
        "oos_gross_sharpe": oos_gross_sharpe,
        "oos_net_sharpe": oos_net_sharpe,
        "maxdd_pct": float(res_net["maxdd_pct"]),
        "trades": int(res_net["trades"]),
        "trades_per_month": trades_per_month,
        "turnover": float(res_net["turnover"]),
        "fees_pct_gross": fees_pct_gross,
        "exposure": float(res_net["exposure_pct"]),
        "hit_rate": float(res_net["hit_rate"]),
        "avg_hold": float(res_net["avg_hold_hours"]),
    }


def aggregate_segments(results: List[Dict]) -> Dict:
    def agg(key, fn=np.nanmean):
        vals = [r.get(key, np.nan) for r in results]
        return float(fn(vals)) if len(vals) else float("nan")

    def agg_std(key):
        vals = [r.get(key, np.nan) for r in results]
        return float(np.nanstd(vals)) if len(vals) else float("nan")

    return {
        "WFV_Avg_OOS_Gross_Sharpe": agg("oos_gross_sharpe"),
        "WFV_Avg_OOS_Net_Sharpe": agg("oos_net_sharpe"),
        "WFV_OOS_Gross_Sharpe_Std": agg_std("oos_gross_sharpe"),
        "WFV_OOS_Net_Sharpe_Std": agg_std("oos_net_sharpe"),
        "WFV_OOS_MaxDD": agg("maxdd_pct"),
        "WFV_OOS_Trades": agg("trades"),
        "WFV_OOS_TradesPerMonth": agg("trades_per_month"),
        "WFV_OOS_Turnover": agg("turnover"),
        "WFV_OOS_FeesPctGrossPnL": agg("fees_pct_gross"),
        "WFV_OOS_Exposure": agg("exposure"),
        "WFV_OOS_HitRate": agg("hit_rate"),
        "WFV_OOS_AvgHold": agg("avg_hold"),
    }


# ================================
# Runner
# ================================


def run_experiment(
    symbols: List[str],
    start: str,
    end: str,
    fees_bps: float = DEFAULT_FEES_BPS,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    out_csv: str = "research/reports/wfv_eth_1h_fee_reduction.csv",
) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        data_by_tf = load_data_all_tf(sym, start=start, end=end)
        for tf in TIMEFRAMES:
            if tf not in data_by_tf:
                continue
            df = data_by_tf[tf]
            print(
                f"Loaded (intersected) {sym} {tf}: {df.index.min()} → {df.index.max()} bars={len(df)}",
                flush=True,
            )
            segs = walk_forward_splits(
                df.index, is_months=12, oos_months=3, roll_months=3
            )
            print(f"Segments: {len(segs)} (need ≥1)", flush=True)
            results = []
            for seg in segs:
                r = evaluate_segment(df, tf, seg, GRID, fees_bps, initial_cash)
                results.append(r)
            agg = aggregate_segments(results)
            row = {"symbol": sym, "tf": tf}
            row.update(agg)
            rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _default_dates() -> Tuple[str, str]:
    return "2022-01-01", "2024-12-31"


def _warn_if_no_ccxt():
    if ccxt is None:
        print(
            "Warning: CCXT not installed. yfinance intraday (30m/1h/4h) history is limited. Install CCXT for fuller OHLCV: pip install ccxt",
            flush=True,
        )


def _parse_args():
    p = argparse.ArgumentParser(description="WFV: 30m vs 1h vs 4h (vol-targeted)")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols like BTCUSDT ETHUSDT",
    )
    p.add_argument("--start", default=_default_dates()[0])
    p.add_argument("--end", default=_default_dates()[1])
    p.add_argument("--fees_bps", type=float, default=DEFAULT_FEES_BPS)
    p.add_argument("--cash", type=float, default=DEFAULT_INITIAL_CASH)
    p.add_argument("--out", default="research/reports/wfv_eth_1h_fee_reduction.csv")
    p.add_argument(
        "--tfs",
        nargs="+",
        choices=["30m", "1h", "4h"],
        help="Optional list of timeframes to evaluate (default: all)",
        default=None,
    )
    p.add_argument(
        "--no_intersect",
        action="store_true",
        help="Do not intersect across timeframes; load each TF independently",
    )
    return p.parse_args()


def main():
    _warn_if_no_ccxt()
    args = _parse_args()
    # Determine which timeframes to run
    tfs = args.tfs or TIMEFRAMES
    rows = []
    if args.no_intersect or len(tfs) == 1:
        # Load each timeframe independently (no cross-TF intersection)
        for sym in args.symbols:
            for tf in tfs:
                try:
                    df_tf = load_data_tf(sym, tf, args.start, args.end)
                except Exception as e:
                    print(f"[WARN] Failed to load {sym} {tf}: {e}")
                    continue
                print(
                    f"Loaded {sym} {tf}: {df_tf.index.min()} → {df_tf.index.max()} bars={len(df_tf)}",
                    flush=True,
                )
                segs = walk_forward_splits(
                    df_tf.index, is_months=12, oos_months=3, roll_months=3
                )
                print(f"Segments: {len(segs)} (need ≥1)", flush=True)
                results = [
                    evaluate_segment(df_tf, tf, seg, GRID, args.fees_bps, args.cash)
                    for seg in segs
                ]
                agg = aggregate_segments(results)
                row = {"symbol": sym, "tf": tf}
                row.update(agg)
                rows.append(row)
        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(out_df)
    else:
        # Keep strict intersection across TFs, but honor selected tfs subset
        out_df = run_experiment(
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            fees_bps=args.fees_bps,
            initial_cash=args.cash,
            out_csv=args.out,
        )
        print(out_df)


if __name__ == "__main__":
    main()
