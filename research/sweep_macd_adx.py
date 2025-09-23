import itertools
import numpy as np
import pandas as pd
import ccxt
import yfinance as yf
from backtesting import Backtest, Strategy

# choose indicator lib
USE_TALIB = False
if USE_TALIB:
    import talib
else:
    import pandas_ta as ta


def _parse_period_days(period: str | int | None) -> int:
    if isinstance(period, (int, float)):
        return int(period)
    if isinstance(period, str) and period.endswith("d"):
        num = period[:-1]
        if num.isdigit():
            return int(num)
    return 730


def _download_yf_hourly(symbol: str, total_days: int) -> pd.DataFrame:
    days = max(1, min(int(total_days), 720))
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
    # --- try coinbase (1h only) ---
    try:
        ex = ccxt.coinbaseadvanced({"enableRateLimit": True, "timeout": 90000})
        raw = ex.fetch_ohlcv(symbol.replace("-", "/"), timeframe="1h", limit=limit)
        df = pd.DataFrame(raw, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
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

    def init(self):
        price_series = pd.Series(self.data.Close).astype(float)
        high_series = pd.Series(self.data.High).astype(float)
        low_series = pd.Series(self.data.Low).astype(float)
        if USE_TALIB:
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
        else:
            mdf = ta.macd(price_series, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            macd_hist_col = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            self.macd_hist = self.I(lambda: mdf[macd_hist_col].values, name="macd_hist")
            adf = ta.adx(high_series, low_series, price_series, length=self.adx_len)
            adx_col = f"ADX_{self.adx_len}"
            self.adx = self.I(lambda: adf[adx_col].values, name="adx")

        rets = price_series.pct_change().fillna(0.0)
        self.realized_vol = self.I(lambda: rets.rolling(self.vol_lb).std().values, name="realized_vol")

    def position_size_scale(self):
        vol = self.realized_vol[-1]
        if vol is None or np.isnan(vol) or vol <= 0:
            return 0.0
        return float(np.clip(self.target_vol / vol, self.size_min, self.size_max))

    def next(self):
        if len(self.macd_hist) < 2:
            return
        if np.isnan(self.adx[-1]) or np.isnan(self.macd_hist[-1]) or np.isnan(self.macd_hist[-2]):
            return
        if self.adx[-1] <= self.adx_threshold:
            return
        crossed_up = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        crossed_down = self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0
        vol_scale = self.position_size_scale()
        if vol_scale <= 0:
            return
        # trade integer μBTC units (size=1 = 0.001 BTC)
        risk_frac = 0.10
        price = float(self.data.Close[-1])
        equity = float(self.equity)
        target_notional = max(0.0, equity * risk_frac * vol_scale)
        units = int(np.floor(target_notional / max(price, 1e-9)))
        if units < 1:
            units = 1
        if crossed_up:
            self.position.close()
            self.buy(size=units)
        elif crossed_down:
            self.position.close()
            self.sell(size=units)


def run_split_backtest(df, params, commission=0.0012):
    k = int(len(df) * 0.7)
    df_is, df_oos = df.iloc[:k], df.iloc[k:]
    bt_is = Backtest(df_is, Strat, cash=10_000, commission=commission, trade_on_close=True, exclusive_orders=True)
    bt_oos = Backtest(df_oos, Strat, cash=10_000, commission=commission, trade_on_close=True, exclusive_orders=True)
    s_is = bt_is.run(**params)
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
        bt_is = Backtest(df_is, Strat, cash=10_000, commission=commission, trade_on_close=True, exclusive_orders=True)
        bt_oos = Backtest(df_oos, Strat, cash=10_000, commission=commission, trade_on_close=True, exclusive_orders=True)
        s_is = bt_is.run(**params)
        s_oos = bt_oos.run(**params)
        scores.append((float(s_is["Sharpe Ratio"]), float(s_oos["Sharpe Ratio"]), int(s_oos["# Trades"])))
    if not scores:
        return float("nan"), 0
    avg_oos_sharpe = float(np.mean([x[1] for x in scores]))
    total_oos_trades = int(np.sum([x[2] for x in scores]))
    return avg_oos_sharpe, total_oos_trades


def main():
    grids = {
        "timeframe": ["1h", "4h"],
        "macd_fast": [8, 12],
        "macd_slow": [24, 26],
        "macd_signal": [9],
        "adx_threshold": [20, 25, 30],
        "vol_lb": [20, 30],
        "target_vol": [0.015, 0.02],
    }
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

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    rows = []

    for sym in symbols:
        df_cache = {}
        for (tf, mf, ms, sig, adx_thr, vol_lb, tgt_vol) in combos:
            if tf not in df_cache:
                df_cache[tf] = load_df(symbol=sym, timeframe=tf, period="1825d", limit=1500)
                df_cache[tf] = to_mbtc(df_cache[tf])  # integer units for backtesting
            df = df_cache[tf]

            params = dict(
                macd_fast=mf,
                macd_slow=ms,
                macd_signal=sig,
                adx_len=14,
                adx_threshold=adx_thr,
                vol_lb=vol_lb,
                target_vol=tgt_vol,
            )

            is_stats, oos_stats = run_split_backtest(df, params, commission=0.0012)

            def pick(s):
                return dict(
                    Sharpe=float(s["Sharpe Ratio"]),
                    MaxDD=float(s["Max. Drawdown [%]"]),
                    Trades=int(s["# Trades"]),
                )

            rows.append(
                {
                    "symbol": sym,
                    "tf": tf,
                    "macd": f"{mf}/{ms}/{sig}",
                    "adx_thr": adx_thr,
                    "vol_lb": vol_lb,
                    "tgt_vol": tgt_vol,
                    "IS_Sharpe": pick(is_stats)["Sharpe"],
                    "IS_MaxDD": pick(is_stats)["MaxDD"],
                    "IS_Trades": pick(is_stats)["Trades"],
                    "OOS_Sharpe": pick(oos_stats)["Sharpe"],
                    "OOS_MaxDD": pick(oos_stats)["MaxDD"],
                    "OOS_Trades": pick(oos_stats)["Trades"],
                }
            )
            print(
                f"Done {sym} {tf} MACD {mf}/{ms}/{sig} ADX>{adx_thr} vol_lb={vol_lb} tgt={tgt_vol}"
            )

    out = pd.DataFrame(rows)

    # Minimum trades filter to avoid spurious Sharpe from tiny samples
    MIN_TRADES = 20
    out_f = out[(out["OOS_Trades"] >= MIN_TRADES) & (out["IS_Trades"] >= MIN_TRADES)].copy()
    out_f = out_f.sort_values(["symbol", "OOS_Sharpe"], ascending=[True, False])

    # Save full + filtered
    out.to_csv("research/sweep_results_full.csv", index=False)
    out_f.to_csv("research/sweep_results_filtered.csv", index=False)

    # Print top 10 per symbol
    for sym in symbols:
        sub = out_f[out_f["symbol"] == sym].head(10)
        if len(sub):
            print(f"\nTOP for {sym}\n", sub.to_string(index=False))

    # Walk-forward validate top 3 per symbol
    wfv_rows = []
    for sym in symbols:
        for _, row in out_f[out_f["symbol"] == sym].head(3).iterrows():
            df_tf = load_df(symbol=sym, timeframe=row["tf"], period="1825d", limit=1500)
            df_tf = to_mbtc(df_tf)
            params = dict(
                macd_fast=int(row["macd"].split("/")[0]),
                macd_slow=int(row["macd"].split("/")[1]),
                macd_signal=int(row["macd"].split("/")[2]),
                adx_len=14,
                adx_threshold=int(row["adx_thr"]),
                vol_lb=int(row["vol_lb"]),
                target_vol=float(row["tgt_vol"]),
            )
            avg_oos, oos_tr = walk_forward_scores(df_tf, params, n_folds=4, commission=0.0012)
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
    wfv = pd.DataFrame(wfv_rows).sort_values(["symbol", "WFV_Avg_OOS_Sharpe"], ascending=[True, False])
    wfv.to_csv("research/sweep_walkforward.csv", index=False)
    if not wfv.empty:
        print("\nWALK-FORWARD SUMMARY\n", wfv.to_string(index=False))
    else:
        print("\nWALK-FORWARD SUMMARY\nNo qualifying configurations for walk-forward validation.")


if __name__ == "__main__":
    main()
