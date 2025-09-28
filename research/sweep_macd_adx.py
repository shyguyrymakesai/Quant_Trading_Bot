import itertools
import os
import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
import ccxt
import yfinance as yf
from backtesting import Backtest, Strategy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use TA-Lib exclusively
USE_TALIB = True
import talib

# Fast/targeted sweep toggle
FAST_MODE = True  # keep True to run the lean targeted sweep described below

# Optional environment toggles for data/filters
PREFER_YF = os.environ.get("SWEEP_PREFER_YF", "0") == "1"
YF_MAX_DAYS = int(os.environ.get("SWEEP_YF_DAYS", "720"))  # clamp handled below
ENV_MIN_TRADES = os.environ.get("SWEEP_MIN_TRADES")  # if set, overrides default
ENV_MIN_OOS_SHARPE = os.environ.get("SWEEP_MIN_OOS_SHARPE")  # if set, overrides default


def _parse_period_days(period: str | int | None) -> int:
    if isinstance(period, (int, float)):
        return int(period)
    if isinstance(period, str) and period.endswith("d"):
        num = period[:-1]
        if num.isdigit():
            return int(num)
    return 730


@contextmanager
def suppress_sortino_runtimewarnings():
    """Suppress 'divide by zero' RuntimeWarnings from backtesting Sortino calc.
    Keeps output clean when there are no negative day returns.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"divide by zero encountered in scalar divide",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            yield


def _download_yf_hourly(symbol: str, total_days: int) -> pd.DataFrame:
    # yfinance limit ~2 years hourly; clamp to env-configured max
    cap = max(1, int(YF_MAX_DAYS))
    days = max(1, min(int(total_days), cap))
    df = yf.download(
        symbol,
        interval="1h",
        period=f"{days}d",
        progress=False,
        auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_df(symbol="BTC-USD", timeframe="1h", period="1825d", limit=1000):
    """
    Try Coinbase Advanced 1h OHLCV; if it fails, fallback to yfinance for ~5y.
    Resample locally to 4h if requested.
    symbol: "BTC-USD" / "ETH-USD" / "SOL-USD"
    """
    # --- prefer yfinance if requested, else try coinbase then fallback ---
    if PREFER_YF:
        total_days = _parse_period_days(period)
        df = _download_yf_hourly(symbol, total_days)
        if df.empty:
            raise RuntimeError(f"No data from yfinance for {symbol}")
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    else:
        try:
            ex = ccxt.coinbaseadvanced({"enableRateLimit": True, "timeout": 90000})
            raw = ex.fetch_ohlcv(symbol.replace("-", "/"), timeframe="1h", limit=limit)
            df = pd.DataFrame(
                raw, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
            df.set_index("Date", inplace=True)
            df = df.astype(float)
        except Exception as e:
            print(f"[WARN] Coinbase fetch failed ({e}); using yfinance {symbol}...")
            total_days = _parse_period_days(period)
            df = _download_yf_hourly(symbol, total_days)
            if df.empty:
                raise RuntimeError(f"No data from yfinance for {symbol}")
            if getattr(df.index, "tz", None) is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    if timeframe == "4h":
        o = df["Open"].resample("4h").first()
        h = df["High"].resample("4h").max()
        l = df["Low"].resample("4h").min()
        c = df["Close"].resample("4h").last()
        v = df["Volume"].resample("4h").sum()
        df = pd.concat([o, h, l, c, v], axis=1).dropna()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df.dropna()


def to_mbtc(df: pd.DataFrame) -> pd.DataFrame:
    """Scale USD/BTC prices → USD per mBTC so 1 unit = 0.001 BTC (integer units)."""
    out = df.copy()
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c] / 1_000.0
    return out


# Cache data per (symbol, timeframe) to avoid repeated loads
_DATA_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def get_df(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    if key not in _DATA_CACHE:
        df = load_df(symbol=symbol, timeframe=timeframe, period="1825d", limit=1500)
        df = to_mbtc(df)  # integer units for backtesting only
        _DATA_CACHE[key] = df
    return _DATA_CACHE[key]


class Strat(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_len = 14
    adx_threshold = 25
    vol_lb = 20
    target_vol = 0.02
    size_min = 0.0
    size_max = 1.0
    # Exit cool-down in bars: after closing a position, wait N bars before re-entering
    exit_cooldown = 0

    def init(self):
        price_series = pd.Series(self.data.Close).astype(float)
        high_series = pd.Series(self.data.High).astype(float)
        low_series = pd.Series(self.data.Low).astype(float)
        macd, macdsig, macdh = talib.MACD(
            price_series.values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        self.macd_hist = self.I(lambda: macdh, name="macd_hist")
        adx = talib.ADX(
            high_series.values,
            low_series.values,
            price_series.values,
            timeperiod=self.adx_len,
        )
        self.adx = self.I(lambda: adx, name="adx")

        rets = price_series.pct_change().fillna(0.0)
        self.realized_vol = self.I(
            lambda: rets.rolling(self.vol_lb).std().values, name="realized_vol"
        )
        # Internal cool-down counter
        self._cooldown_ctr = 0

    def position_size_scale(self):
        vol = self.realized_vol[-1]
        if vol is None or np.isnan(vol) or vol <= 0:
            return 0.0
        return float(np.clip(self.target_vol / vol, self.size_min, self.size_max))

    def next(self):
        # guard early NaNs
        if len(self.macd_hist) < 2:
            return
        if np.isnan(self.macd_hist[-1]) or np.isnan(self.macd_hist[-2]):
            return
        # Only enforce ADX if threshold > 0
        if self.adx_threshold > 0:
            if np.isnan(self.adx[-1]) or self.adx[-1] <= self.adx_threshold:
                return

        crossed_up = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        crossed_down = self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0

        vol_scale = self.position_size_scale()
        if vol_scale <= 0:
            return

        # integer μBTC units (backtesting only)
        risk_frac = 0.10
        price = float(self.data.Close[-1])
        equity = float(self.equity)
        target_notional = max(0.0, equity * risk_frac * vol_scale)
        units = int(np.floor(target_notional / max(price, 1e-9)))
        if units < 1:
            units = 1

        # decrement cool-down if active
        if self._cooldown_ctr > 0:
            self._cooldown_ctr -= 1

        # If signal flips, always close first and start cool-down
        if crossed_up and self.position.is_short:
            self.position.close()
            self._cooldown_ctr = int(self.exit_cooldown)
            return  # no immediate re-entry on the same bar
        if crossed_down and self.position.is_long:
            self.position.close()
            self._cooldown_ctr = int(self.exit_cooldown)
            return  # no immediate re-entry on the same bar

        # Respect cool-down before new entries
        if self._cooldown_ctr > 0:
            return

        # Entries from flat or aligned with signal
        if crossed_up and not self.position:
            self.buy(size=units)
        elif crossed_down and not self.position:
            self.sell(size=units)


def eval_combo(args):
    sym, tf, macd_tuple, adx_thr, vol_lb, tgt_vol, exit_cd = args
    mf, ms, sig = macd_tuple
    df = get_df(sym, tf)
    params = dict(
        macd_fast=mf,
        macd_slow=ms,
        macd_signal=sig,
        adx_len=14,
        adx_threshold=adx_thr,
        vol_lb=vol_lb,
        target_vol=tgt_vol,
        exit_cooldown=exit_cd,
    )
    # Commission approximates maker-only entries + market exits + light slippage
    is_stats, oos_stats = run_split_backtest(df, params, commission=0.0012)
    return {
        "symbol": sym,
        "tf": tf,
        "macd": f"{mf}/{ms}/{sig}",
        "adx_thr": adx_thr,
        "vol_lb": vol_lb,
        "tgt_vol": tgt_vol,
        "exit_cd": exit_cd,
        "IS_Sharpe": float(is_stats["Sharpe Ratio"]),
        "IS_MaxDD": float(is_stats["Max. Drawdown [%]"]),
        "IS_Trades": int(is_stats["# Trades"]),
        "OOS_Sharpe": float(oos_stats["Sharpe Ratio"]),
        "OOS_MaxDD": float(oos_stats["Max. Drawdown [%]"]),
        "OOS_Trades": int(oos_stats["# Trades"]),
    }


def run_split_backtest(df, params, commission=0.0012):
    k = int(len(df) * 0.7)
    df_is, df_oos = df.iloc[:k], df.iloc[k:]
    bt_is = Backtest(
        df_is,
        Strat,
        cash=10_000,
        commission=commission,
        trade_on_close=True,  # approximates market timing; maker-only entries cannot be modeled per-order here
        exclusive_orders=True,
    )
    bt_oos = Backtest(
        df_oos,
        Strat,
        cash=10_000,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True,
    )
    with suppress_sortino_runtimewarnings():
        s_is = bt_is.run(**params)
    with suppress_sortino_runtimewarnings():
        s_oos = bt_oos.run(**params)
    return s_is, s_oos


def walk_forward_scores(df, params, n_folds=4, commission=0.0012):
    scores = []
    N = len(df)
    # anchored expanding IS with equal-ish OOS slices
    cut_points = [int(N * (i / (n_folds + 1))) for i in range(1, n_folds + 1)]
    for i, cp in enumerate(cut_points):
        oos_len = int((N - cp) / (n_folds + 1 - i))
        df_is, df_oos = df.iloc[:cp], df.iloc[cp : cp + oos_len]
        if len(df_is) < 10 or len(df_oos) < 10:
            continue
        bt_is = Backtest(
            df_is,
            Strat,
            cash=10_000,
            commission=commission,
            trade_on_close=True,
            exclusive_orders=True,
        )
        bt_oos = Backtest(
            df_oos,
            Strat,
            cash=10_000,
            commission=commission,
            trade_on_close=True,
            exclusive_orders=True,
        )
        with suppress_sortino_runtimewarnings():
            s_is = bt_is.run(**params)
        with suppress_sortino_runtimewarnings():
            s_oos = bt_oos.run(**params)
        scores.append(
            (
                float(s_is["Sharpe Ratio"]),
                float(s_oos["Sharpe Ratio"]),
                int(s_oos["# Trades"]),
            )
        )
    if not scores:
        return float("nan"), 0
    avg_oos_sharpe = float(np.mean([x[1] for x in scores]))
    total_oos_trades = int(np.sum([x[2] for x in scores]))
    return avg_oos_sharpe, total_oos_trades


def main():
    # Grids: slim in FAST_MODE, full otherwise
    if FAST_MODE:
        # Targeted sweep per request:
        # Symbols/TF: BTC-USD, ETH-USD, 1h
        # MACD sets: (8,24,9), (5,24,9), (8,24,5)
        # ADX threshold: 20, 25, 30
        # Vol lookback: 20, 30
        # Target vol: 0.015, 0.020
        # Exit cool-down (bars): 0, 4, 6
        grids = {
            "timeframe": ["1h"],
            "macd_sets": [(8, 24, 9), (5, 24, 9), (8, 24, 5)],
            "adx_threshold": [20, 25, 30],
            "vol_lb": [20, 30],
            "target_vol": [0.015, 0.020],
            "exit_cd": [0, 4, 6],
        }
    else:
        grids = {
            "timeframe": ["1h", "4h"],
            "macd_fast": [5, 6, 8, 10, 12],
            "macd_slow": [19, 24, 26, 30, 35],
            "macd_signal": [5, 9, 10],
            "adx_threshold": [0, 5, 10, 15, 20, 25],
            "vol_lb": [10, 20, 30, 40],
            "target_vol": [0.02, 0.03, 0.05, 0.10],
        }

    if FAST_MODE:
        combos = list(
            itertools.product(
                grids["timeframe"],
                grids["macd_sets"],
                grids["adx_threshold"],
                grids["vol_lb"],
                grids["target_vol"],
                grids["exit_cd"],
            )
        )
    else:
        combos = list(
            itertools.product(
                grids["timeframe"],
                grids["macd_fast"],
                grids["macd_slow"],
                grids["macd_signal"],
                grids["adx_threshold"],
                grids["vol_lb"],
                grids["target_vol"],
            )
        )

    symbols = (
        ["BTC-USD", "ETH-USD"]
        if FAST_MODE
        else ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "ADA-USD"]
    )

    # Prepare jobs for parallel execution
    jobs = []
    for sym in symbols:
        if FAST_MODE:
            for tf, macd_set, adx_thr, vol_lb, tgt_vol, exit_cd in combos:
                jobs.append((sym, tf, macd_set, adx_thr, vol_lb, tgt_vol, exit_cd))
        else:
            for tf, mf, ms, sig, adx_thr, vol_lb, tgt_vol in combos:
                jobs.append((sym, tf, (mf, ms, sig), adx_thr, vol_lb, tgt_vol, 0))

    results = []
    workers = (
        max(1, (os.cpu_count() or 4) - 1)
        if FAST_MODE
        else max(1, (os.cpu_count() or 4) - 2)
    )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(eval_combo, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(
                f"Done {r['symbol']} {r['tf']} MACD {r['macd']} ADX>{r['adx_thr']} vol_lb={r['vol_lb']} tgt={r['tgt_vol']} cd={r['exit_cd']}"
            )

    out = pd.DataFrame(results)

    # Minimum trades filter to avoid spurious Sharpe from tiny samples
    # Filters (overridable via env vars for quick exploration)
    MIN_TRADES = (
        int(ENV_MIN_TRADES)
        if (ENV_MIN_TRADES and ENV_MIN_TRADES.isdigit())
        else (20 if FAST_MODE else 30)
    )  # require >=X IS and OOS trades
    try:
        MIN_OOS_SHARPE = float(ENV_MIN_OOS_SHARPE) if ENV_MIN_OOS_SHARPE else 0.5
    except ValueError:
        MIN_OOS_SHARPE = 0.5
    out_f = out[
        (out["OOS_Trades"] >= MIN_TRADES) & (out["IS_Trades"] >= MIN_TRADES)
    ].copy()
    if out_f.empty:
        # If empty on first pass, temporarily lower to 20 to inspect the frontier
        MIN_TRADES = 20
        out_f = out[
            (out["OOS_Trades"] >= MIN_TRADES) & (out["IS_Trades"] >= MIN_TRADES)
        ].copy()
    out_f = out_f.sort_values(["symbol", "OOS_Sharpe"], ascending=[True, False])

    # Save full + filtered next to this script
    base = Path(__file__).resolve().parent
    (
        (base / "sweep_results_full.csv").write_text("")
        if out.empty
        else out.to_csv(base / "sweep_results_full.csv", index=False)
    )
    (
        (base / "sweep_results_filtered.csv").write_text("")
        if out_f.empty
        else out_f.to_csv(base / "sweep_results_filtered.csv", index=False)
    )

    # Print top 10 per symbol
    for sym in symbols:
        sub = out_f[out_f["symbol"] == sym].head(10)
        if len(sub):
            print(f"\nTOP for {sym}\n", sub.to_string(index=False))

    # Walk-forward validate small top-K per symbol
    n_folds = 2 if FAST_MODE else 4
    TOP_K = 3 if FAST_MODE else 5
    wfv_rows = []
    for sym in symbols:
        sub = out_f[
            (out_f["symbol"] == sym) & (out_f["OOS_Sharpe"] >= MIN_OOS_SHARPE)
        ].head(TOP_K)
        for _, row in sub.iterrows():
            df_tf = get_df(symbol=sym, timeframe=row["tf"])
            params = dict(
                macd_fast=int(row["macd"].split("/")[0]),
                macd_slow=int(row["macd"].split("/")[1]),
                macd_signal=int(row["macd"].split("/")[2]),
                adx_len=14,
                adx_threshold=int(row["adx_thr"]),
                vol_lb=int(row["vol_lb"]),
                target_vol=float(row["tgt_vol"]),
                exit_cooldown=int(row.get("exit_cd", 0)),
            )
            avg_oos, oos_tr = walk_forward_scores(
                df_tf, params, n_folds=n_folds, commission=0.0012
            )
            wfv_rows.append(
                {
                    "symbol": sym,
                    "tf": row["tf"],
                    "macd": row["macd"],
                    "adx_thr": row["adx_thr"],
                    "vol_lb": row["vol_lb"],
                    "tgt_vol": row["tgt_vol"],
                    "WFV_Avg_OOS_Sharpe": avg_oos,
                    "WFV_OOS_Trades": oos_tr,
                }
            )
    if wfv_rows:
        wfv = pd.DataFrame(wfv_rows)
        if all(c in wfv.columns for c in ["symbol", "WFV_Avg_OOS_Sharpe"]):
            wfv = wfv.sort_values(
                ["symbol", "WFV_Avg_OOS_Sharpe"], ascending=[True, False]
            )
        wfv.to_csv(
            Path(__file__).resolve().parent / "sweep_walkforward.csv", index=False
        )
        if not wfv.empty:
            print("\nWALK-FORWARD SUMMARY\n", wfv.to_string(index=False))
        else:
            print(
                "\nWALK-FORWARD SUMMARY\nNo qualifying configurations for walk-forward validation."
            )
    else:
        (Path(__file__).resolve().parent / "sweep_walkforward.csv").write_text("")
        print(
            "\nWALK-FORWARD SUMMARY\nNo qualifying configurations for walk-forward validation."
        )


if __name__ == "__main__":
    main()
