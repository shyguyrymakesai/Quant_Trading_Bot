import itertools
import numpy as np
import pandas as pd
import ccxt
from backtesting import Backtest, Strategy
import pandas_ta as ta

USE_TALIB = False
if USE_TALIB:
    import talib

def load_df(symbol="BTC/USDT", timeframe="1h", limit=3000, exchange="binance"):
    if exchange.lower() == "binance":
        ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    else:
        ex = ccxt.coinbaseadvanced({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["Date","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    return df.astype(float)

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
        price = self.data.Close
        if USE_TALIB:
            macd, macdsignal, macdhist = talib.MACD(
                price.values, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
            )
            self.macd_hist = self.I(lambda: macdhist, name="macd_hist")
            adx = talib.ADX(self.data.High.values, self.data.Low.values, price.values, timeperiod=self.adx_len)
            self.adx = self.I(lambda: adx, name="adx")
        else:
            macd_df = ta.macd(price, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            adx_df = ta.adx(high=self.data.High, low=self.data.Low, close=price, length=self.adx_len)
            self.macd_hist = self.I(lambda: macd_df.iloc[:,2].values, name="macd_hist")
            self.adx = self.I(lambda: adx_df.iloc[:,0].values, name="adx")

        rets = price.pct_change().fillna(0.0)
        self.realized_vol = self.I(lambda: rets.rolling(self.vol_lb).std().values, name="realized_vol")

    def position_size_scale(self):
        vol = self.realized_vol[-1]
        if vol is None or np.isnan(vol) or vol <= 0:
            return 0.0
        return float(np.clip(self.target_vol / vol, self.size_min, self.size_max))

    def next(self):
        if self.adx[-1] <= self.adx_threshold:
            return
        size_scale = self.position_size_scale()
        if size_scale <= 0:
            return
        crossed_up = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        crossed_down = self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0
        if crossed_up:
            self.position.close()
            self.buy(size=size_scale)
        elif crossed_down:
            self.position.close()
            self.sell(size=size_scale)

def run_split_backtest(df, params, commission=0.0012):
    k = int(len(df) * 0.7)
    df_is, df_oos = df.iloc[:k], df.iloc[k:]

    def run(df_part):
        bt = Backtest(df_part, Strat, cash=10_000, commission=commission, trade_on_close=True, exclusive_orders=True)
        return bt.run(**params)

    stats_is = run(df_is)
    stats_oos = run(df_oos)
    return stats_is, stats_oos

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

    combos = list(itertools.product(
        grids["timeframe"], grids["macd_fast"], grids["macd_slow"],
        grids["macd_signal"], grids["adx_threshold"], grids["vol_lb"], grids["target_vol"]
    ))

    rows = []
    for (tf, mf, ms, sig, adx_thr, vol_lb, tgt_vol) in combos:
        df = load_df("BTC/USDT", timeframe=tf, limit=2500, exchange="binance")
        # patch params into class via **kwargs
        params = dict(
            macd_fast=mf, macd_slow=ms, macd_signal=sig,
            adx_len=14, adx_threshold=adx_thr,
            vol_lb=vol_lb, target_vol=tgt_vol
        )
        is_stats, oos_stats = run_split_backtest(df, params)

        def pick(s):
            return dict(
                Return=float(s["Return [%]"]),
                Sharpe=float(s["Sharpe Ratio"]),
                MaxDD=float(s["Max. Drawdown [%]"]),
                Trades=int(s["# Trades"]),
            )

        rows.append({
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
        })
        print(f"Done {tf}, MACD {mf}/{ms}/{sig}, ADX>{adx_thr}, vol_lb={vol_lb}, tgt={tgt_vol}")

    out = pd.DataFrame(rows).sort_values(["OOS_Sharpe"], ascending=False)
    print("\nTOP RESULTS (sorted by OOS Sharpe)\n", out.head(15).to_string(index=False))
    out.to_csv("research/sweep_results.csv", index=False)
    print("\nSaved: research/sweep_results.csv")

if __name__ == "__main__":
    main()
