# Crypto Momentum Bot (MACD + ADX + Volatility Targeting)

A research-backed crypto trading bot that implements a **MACD signal filtered by ADX > 25**, sized by **volatility-targeting (Moreira–Muir)**, and designed for **paper trading first** (Binance Spot Testnet or Coinbase Advanced sandbox). Extension: optional sentiment gating.

## Stack
- **Python 3.10+**
- **CCXT** for exchange access
- **Backtesting.py** for quick backtests (research) — can swap to vectorbt later
- **pandas / pandas-ta / numpy**
- **FastAPI** + **Uvicorn** for a tiny admin API
- **APScheduler** for scheduling
- **SQLite (aiosqlite)** via SQLAlchemy for lightweight state
- **Docker** for packaging

## Quickstart
1. **Create `.env`** (or use `config/config.yaml`) and set keys for your chosen exchange testnet.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a one-off backtest (sanity check):
   ```bash
   python research/backtest_macd_adx.py
   ```
4. Start paper-trading bot (hourly by default):
   ```bash
   python scripts/run_bot.py
   ```
5. (Optional) Launch the admin API:
   ```bash
   uvicorn src.app:app --reload --port 8000
   ```

> **Safety**: Start with **paper trading**. When flipping to live, keep **position sizes tiny** (this project is designed for ~$30/mo).

## Project Structure
```
├─ config/
│  └─ config.example.yaml
├─ research/
│  └─ backtest_macd_adx.py
├─ scripts/
│  └─ run_bot.py
├─ src/
│  ├─ app.py                # FastAPI admin (status, trades, metrics)
│  ├─ config.py             # Loads YAML/env into Pydantic Settings
│  ├─ db.py                 # SQLAlchemy models + session helpers
│  ├─ data_adapter.py       # CCXT historical & recent candles
│  ├─ execution_adapter.py  # Order placement (paper/live)
│  ├─ risk_manager.py       # Daily loss cap, max position, etc.
│  ├─ signal_engine.py      # MACD, ADX, vol targeting orchestration
│  ├─ scheduler.py          # APScheduler job wiring
│  └─ strategies/macd_adx.py# Core indicator math & signal logic
├─ docker/
│  └─ Dockerfile
├─ requirements.txt
└─ README.md
```

## Milestones (3-week plan checklist)
- [ ] Week 1: Repo + data pull
- [ ] Week 1: MACD/ADX backtests with costs & OOS split
- [ ] Week 1: Volatility targeting & parameter sweep
- [ ] Week 2: Paper-trade skeleton w/ CCXT testnet
- [ ] Week 2: Scheduler + logging + audit DB
- [ ] Week 2: Dockerize
- [ ] Week 3: Risk guards + admin API
- [ ] Week 3: Sentiment v1 gating
- [ ] Week 3: Deploy to $5 VPS (paper), optional live

## Notes
- Fees/slippage **matter**. Set realistic taker fees in research/backtest_macd_adx.py.
- Vol targeting: scale notional by inverse realized volatility (e.g., 20-day ATR or return std).
