# Crypto Momentum Bot (MACD + ADX + Volatility Targeting)

Research-backed crypto momentum bot using TA-Lib MACD with optional ADX gating and volatility-targeted sizing. Built for paper trading first, with clear risk controls and a path to live trading.

Highlights
- TA-Lib-only indicators (no pandas_ta)
- Coinbase Advanced by default; Binance wiring available
- Local 4h resampling from 1h candles (Coinbase doesn’t serve 4h directly)
- Volatility targeting (Moreira–Muir) for position sizing
- Spread guard, commission/slippage modeling, and max position caps
- Robust research tools: fast parameter sweeps, walk-forward validation, CSV outputs
- Paper-trade results logged to both SQLite and CSV for analysis

## Stack
- Python 3.11 (works on 3.10+ as well)
- ccxt for exchange access (coinbaseadvanced)
- TA-Lib for indicators, pandas/numpy for data
- backtesting.py for research
- APScheduler for scheduling
- aiosqlite for lightweight state (bot_state.sqlite3)
- PyYAML + pydantic-settings for configuration

## Project Structure
```
config/
   config.example.yaml
   config.yaml              # your runtime config
   params.yaml              # per-symbol robust params
research/
   backtest_macd_adx.py     # sanity backtest
   backtest_macd_adx_talib.py
   sweep_macd_adx.py        # fast/full sweeps + WFV
scripts/
   run_bot.py               # tick once then hourly schedule
src/
   app.py                   # optional API (status/trades)
   config.py                # loads YAML/env to Settings
   data_adapter.py          # OHLCV fetch (CCXT, yfinance fallback)
   db.py                    # async sqlite logging (trades/metrics)
   execution_adapter.py     # order placement; spread guard; sizing
   position_sizing.py       # exchange-aware qty/price rounding
   risk_manager.py          # daily loss guard scaffold
   scheduler.py             # orchestrates periodic ticks
   signal_engine.py         # indicators, signals, vol sizing
   csv_logger.py            # paper trades CSV logger
   strategies/
      macd_adx.py
docker/
   Dockerfile
requirements.txt
README.md
```

## Configuration

Two YAML files drive behavior:
- `config/config.yaml` — runtime settings
- `config/params.yaml` — per-symbol robust params (overrides)

Key fields in `config.yaml`:
- mode: paper | live
- exchange: coinbase | binance
- symbol: e.g., BTC/USD
- timeframe: 1h (4h via local resampling in research)
- risk:
   - commission (e.g., 0.0012)
   - risk_frac (fraction of equity for a new entry before vol-scaling)
   - max_pos_frac (cap per-asset)
   - daily_loss_limit (placeholder guard)
   - spread_bps_limit (skip if wider)
   - slippage_bps (for modeling)
- api_keys:
   - coinbase: key, secret, passphrase
   - binance: key, secret

Per-symbol overrides `params.yaml` (example):
```
BTC-USD:
   timeframe: 1h
   macd: [8, 24, 9]
   adx_threshold: 20
   vol_lb: 20
   target_vol: 0.02
ETH-USD:
   timeframe: 1h
   macd: [8, 24, 5]
   adx_threshold: 20
   vol_lb: 30
   target_vol: 0.02
```

Environment variables (optional):
- CONFIG_PATH — absolute path to config.yaml
- PARAMS_PATH — absolute path to params.yaml
- BOT_DB_PATH — path to sqlite file (default bot_state.sqlite3)

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

TA-Lib note: wheels are available for most modern Python versions on Windows. If you hit install issues, try updating pip and setuptools first.

## Quick sanity checks

- Backtest sample strategy:
```powershell
python research\backtest_macd_adx_talib.py
```

- Fast parameter sweep (saves CSVs in research/):
```powershell
python research\sweep_macd_adx.py
```
Outputs:
- research/sweep_results_full.csv
- research/sweep_results_filtered.csv
- (WFV printed summaries; can be extended to save CSV)

## Paper trading

Ensure `config/config.yaml` has `mode: "paper"` and `exchange: "coinbase"` (default). Paper mode avoids authenticated endpoints and will:
- Fetch 1h OHLCV from Coinbase; fallback to yfinance if needed
- Compute indicators (MACD with optional ADX gating)
- Size via volatility targeting
- Apply spread guard (paper uses a conservative assumed spread)
- “Place” a paper order (no live call), log to DB and CSV

Run a tick and start scheduler (hourly):
```powershell
python scripts\run_bot.py
```

Paper logs:
- SQLite trade log: `bot_state.sqlite3` (trades table)
- CSV trade log: `logs/paper_trades.csv`

Tip: A single immediate tick runs before the scheduler starts, so you’ll see HOLD or an ORDER right away.

## Live trading (when ready)

1. Add your Coinbase Advanced API credentials to `config/config.yaml`.
2. Set `mode: "live"`.
3. Keep `risk_frac` small and verify exchange limits/precision with tiny orders first.

Live mode differences:
- Uses real order book to enforce spread guard
- Authenticates and can place post-only limit orders
- Position sizing uses exchange precision/min limits

## How the strategy works

- Indicators: TA-Lib MACD (fast/slow/signal), optional ADX filter (threshold > 0 enables)
- Signals: Crosses of MACD histogram over/under zero
- Sizing: Volatility targeting — target_notional ~ equity * risk_frac * (target_vol / realized_vol), capped by max_pos_frac
- 1h bars; 4h available via local resampling in research

## Research suite

- `research/backtest_macd_adx_talib.py`: Sanity backtests with TA-Lib; suppresses divide-by-zero warnings around Sortino.
- `research/sweep_macd_adx.py`: Fast grid sweeps with caching, parallelization, and early pruning; saves CSVs and prints top combinations; includes light walk-forward validation.

Configurable toggles in `sweep_macd_adx.py`:
- FAST_MODE = True/False — smaller grids, fewer folds, top-K pruning
- Grids for MACD/ADX/vol target/lookback can be expanded

## Data and symbols

- Live/paper: CCXT symbols use slashes, e.g., `BTC/USD`
- Per-symbol params: YAML keys use dashes, e.g., `BTC-USD` (handled automatically)
- Coinbase 4h is resampled locally from 1h in research scripts
- yfinance fallback supports `BTC-USD`/`ETH-USD`, etc.

## Outputs and logs

- SQLite: `bot_state.sqlite3` — table `trades` includes ts, symbol, side, qty, price, fee, rationale
- CSV: `logs/paper_trades.csv` — timestamped paper orders with size, price, realized_vol, size_scale, spread_bps, rationale
- Research CSVs in `research/`

## Troubleshooting

- Binance 451 (restricted location): Use Coinbase unless you’re in a supported region.
- Coinbase 401 during market load in paper mode: We avoid authenticated market loads in paper; ensure `mode: "paper"`.
- TA-Lib install issues on Windows: update pip/setuptools; or install a prebuilt wheel compatible with your Python version.
- Empty CSVs after sweeps: Relax trade-count filters or expand the grids; ensure yfinance fallback returned data.

## Roadmap

- Daily loss guard wired to real PnL and equity
- Public market data endpoint for more realistic paper spreads
- Extend walk-forward validation outputs to CSV
- Optional loop runner for faster-than-hourly testing

## License

For research and educational purposes only. Use at your own risk.
