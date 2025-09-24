# Crypto Momentum Bot (MACD + ADX + Volatility Targeting)

A 30-minute MACD+ADX momentum daemon with volatility targeting and idempotent
execution. The scheduler fires on :00/:30 in `America/Indiana/Indianapolis`,
pulls fresh OHLCV, validates parity against the research engine, sizes risk via
Moreira–Muir style vol targeting, and routes paper orders through a broker
adapter.

This repository still ships the research utilities (fast sweeps, backtests), but
now includes an ops-ready paper trading stack:

- **`src/bot_daemon.py`** — APScheduler daemon with circuit breaker,
  parity check, signal-safe shutdown, and structured logging.
- **`src/strategy_macd.py`** — Indicator + signal engine using TA-Lib MACD/ADX
  and volatility-targeted sizing.
- **`src/broker.py`** — Paper broker (immediate fills, journal integration)
  with a thin ccxt-backed live skeleton.
- **`src/state.py`** — JSON-backed journal storing run locks, orders,
  positions, cycle metadata, and market marks.
- **`config/.env`**-style configuration via `src/config.py` with
  `ENV=paper|live`, DRY_RUN toggle, and per-symbol parameters.
- Docker + docker-compose packaging for hands-free deploys.

## 1. Configuration

The daemon reads both YAML (optional legacy) and environment variables. The
simplest path is to copy `.env.example` to `.env` and tweak values:

```bash
cp .env.example .env
```

Key settings:

| Variable | Description |
| --- | --- |
| `ENV` | `paper` or `live` (determines broker wiring). |
| `DRY_RUN` | `1` to log intended orders without sending. |
| `SYMBOLS` | Comma-separated CCXT symbols (`BTC/USDT`). |
| `TIMEFRAME` | Candle timeframe (default `30m`). |
| `TZ` | Scheduler timezone (default `America/Indiana/Indianapolis`). |
| `MACD_FAST/SLOW/SIGNAL` | MACD parameters (defaults `8/24/9`). |
| `ADX_THRESHOLD` | ADX gating (default `20`). |
| `VOL_LOOKBACK` | Volatility lookback (default `20` bars). |
| `TARGET_DAILY_VOL` | Daily vol target (default `0.02`). |
| `STARTING_EQUITY` | Paper account starting cash. |
| `MAX_LEVERAGE` | Multiplier applied to target notional (default `1`). |
| `STATE_FILE` | Path to JSON state journal. |
| `LOG_DIR` | Directory for rotating daemon logs. |
| API keys | `PAPER_*` and `LIVE_*` credentials/endpoints. |

Per-symbol overrides can still be supplied via `config/params.yaml` and are
merged automatically.

## 2. Running the daemon locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.bot_daemon
```

The scheduler waits `PROCESSING_LAG_SECONDS` (default 20s) after :00/:30, then
runs one cycle per symbol. Logs stream to stdout and to `logs/bot_daemon.log`.
State (orders, run locks, account) persists to `state/bot_state.json` by
default.

### Dry-run vs paper vs live

| Mode | Toggle | Behaviour |
| --- | --- | --- |
| Dry run | `DRY_RUN=1` | Log signals + intended orders, no state changes to cash/positions. |
| Paper | `ENV=paper`, `DRY_RUN=0` | Immediate paper fills via `PaperBroker`, state journal updated. |
| Live | `ENV=live`, `DRY_RUN=0` | ccxt-backed broker skeleton (Binance/Coinbase). Extend with venue specifics before use. |

> **Safety** — the daemon is idempotent per candle via `StateStore.candle_lock`
and will skip duplicate processing. `max_instances=1`, `coalesce=True`, and
`misfire_grace_time` guard against overlapping runs.

## 3. Ops playbook

- **Logs** — inspect `logs/bot_daemon.log` (rotating handler) or container logs.
- **State/journal** — `state/bot_state.json` tracks account, positions,
  last orders, run locks, and per-cycle payloads.
- **Circuit breaker** — after `MAX_API_ERRORS` consecutive failures the
  scheduler pauses; clear by restarting the process (state persists).
- **Manual unlock** — delete the relevant symbol entry in
  `state/bot_state.json` under `run_lock`/`last_processed` if you need to
  reprocess a candle (ensure the daemon is stopped first).
- **Flip to dry run** — set `DRY_RUN=1` (env var or `.env`), restart daemon.
- **Change symbols/params** — edit `.env` (or YAML), restart; no code changes.

Backtest parity runs each cycle (`compute_signal_history` vs legacy
`signal_engine`), logging discrepancies for manual review.

## 4. Docker deployment

A slim container is provided. Build + run via docker-compose:

```bash
docker compose build
docker compose up -d
```

The compose file mounts `./logs` and `./state` for persistence and reads
environment variables from `.env`.

## 5. Testing

Unit tests cover indicator math, sizing, broker idempotency, and state locking.
Run them with:

```bash
pytest
```

## 6. Research toolkit (unchanged)

The `research/` folder retains the historical MACD/ADX sweep + backtesting
scripts, and `src/` still exposes reusable adapters for data and execution. Use
those utilities for extended analysis and parameter generation.

---

### File map (new additions)

- `src/bot_daemon.py` — daemon entrypoint.
- `src/strategy_macd.py` — indicator/signal logic.
- `src/broker.py` — broker adapters.
- `src/state.py` — persistent journal + run locks.
- `Dockerfile`, `docker-compose.yml` — container runtime.
- `.env.example` — starter configuration.

For detailed code structure see inline docstrings and comments.
