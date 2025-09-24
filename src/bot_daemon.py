from __future__ import annotations

import asyncio
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ccxt  # type: ignore
import pandas as pd
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from .broker import BrokerError, BrokerFactory
from .config import settings, Settings
from .state import StateStore
from .strategy_macd import (
    Signal,
    SignalResult,
    StrategyParams,
    compute_indicators,
    compute_signal,
    compute_signal_history,
)

try:  # optional legacy comparison
    from .signal_engine import compute_indicators as legacy_indicators
    from .signal_engine import last_signal as legacy_last_signal
except Exception:  # pragma: no cover - optional dependency
    legacy_indicators = None
    legacy_last_signal = None


logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, cfg: Settings) -> None:
        provider = (cfg.data_provider or cfg.exchange or "binance").lower()
        if provider == "binance":
            self.exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        elif provider in {"coinbase", "coinbaseadvanced"}:
            self.exchange = ccxt.coinbaseadvanced({"enableRateLimit": True})
        else:
            raise ValueError(f"Unsupported data provider: {provider}")
        self.cfg = cfg
        self.exchange.load_markets()

    def fetch(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        market_symbol = self.cfg.as_ccxt_symbol(symbol)
        return self.exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, limit=limit)


def _setup_logging(cfg: Settings) -> None:
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "bot_daemon.log"
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    console = logging.StreamHandler()
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        handlers=[handler, console],
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_timeframe_minutes(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 60)
    if tf.endswith("d"):
        return int(float(tf[:-1]) * 1440)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _expected_candle_close(ts: datetime, bar_minutes: int) -> datetime:
    total_minutes = ts.hour * 60 + ts.minute
    bucket = total_minutes // bar_minutes
    close_minutes = bucket * bar_minutes
    hour = close_minutes // 60
    minute = close_minutes % 60
    truncated = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if truncated > ts:
        truncated -= timedelta(minutes=bar_minutes)
    return truncated


def _make_order_id(symbol: str, candle_key: str, side: str, qty: float) -> str:
    base = f"{symbol}|{candle_key}|{side}|{qty:.8f}"
    return hashlib.sha1(base.encode()).hexdigest()[:20]


def _extract_equity(account: Dict[str, object], default: float, quote_currency: str) -> float:
    if not isinstance(account, dict):
        return default
    if "equity" in account:
        try:
            return float(account.get("equity", default))
        except (TypeError, ValueError):
            return default
    if "total" in account and isinstance(account["total"], dict):
        totals = account["total"]
        quote = quote_currency.replace("/", "")
        if quote in totals:
            try:
                return float(totals[quote])
            except (TypeError, ValueError):
                pass
    return default


class BotDaemon:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.timezone_name)
        self.state = StateStore(cfg.state_file, initial_cash=cfg.start_cash)
        self.broker = BrokerFactory.create(self.state, cfg)
        self.data = DataFetcher(cfg)
        self.scheduler = AsyncIOScheduler(timezone=self.tz)
        self.shutdown_event = asyncio.Event()
        self.api_error_count = 0
        self.circuit_breaker = False
        self.bar_minutes = _parse_timeframe_minutes(cfg.timeframe)
        self.params = StrategyParams(
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
            macd_signal=cfg.macd_signal,
            adx_length=cfg.adx_len,
            adx_threshold=cfg.adx_threshold,
            vol_lookback=cfg.vol_lookback,
            target_daily_vol=cfg.vol_target,
            min_fraction=cfg.min_size,
            max_fraction=cfg.max_size,
            bar_minutes=self.bar_minutes,
        )

    async def start(self) -> None:
        _setup_logging(self.cfg)
        self._setup_signal_handlers()
        trigger = CronTrigger(minute="0,30", second=0, timezone=self.tz)
        self.scheduler.add_job(
            self.run_cycle,
            trigger=trigger,
            max_instances=self.cfg.max_instances,
            coalesce=True,
            misfire_grace_time=self.cfg.misfire_grace_time,
        )
        self.scheduler.add_listener(self._job_listener, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
        self.scheduler.start()
        logger.info(
            "Bot daemon started in %s mode (dry_run=%s) for symbols %s",
            self.cfg.env,
            self.cfg.dry_run,
            ", ".join(self.cfg.symbol_universe),
        )
        await self.shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        logger.info("Bot daemon stopped.")

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown(s)))
            except NotImplementedError:  # pragma: no cover - Windows
                signal.signal(sig, lambda *_: asyncio.create_task(self._shutdown(sig)))

    async def _shutdown(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s; shutting down.", sig.name)
        self.shutdown_event.set()

    def _job_listener(self, event) -> None:  # pragma: no cover - scheduler callback
        if event.exception:
            logger.error("Job raised exception: %s", event.exception)
        else:
            logger.warning("Job event: %s", event)

    async def run_cycle(self) -> None:
        if self.circuit_breaker:
            logger.warning("Circuit breaker active; skipping cycle.")
            return
        start = datetime.now(tz=self.tz)
        logger.info("Cycle start at %s", start.isoformat())
        await asyncio.sleep(max(0, self.cfg.processing_lag_seconds))
        errors = 0
        for symbol in self.cfg.symbol_universe:
            try:
                await self._process_symbol(symbol, start)
            except Exception as exc:
                errors += 1
                self.api_error_count += 1
                logger.exception("Error processing %s: %s", symbol, exc)
        if errors == 0:
            self.api_error_count = 0
        if self.api_error_count >= self.cfg.max_api_errors:
            logger.error("Circuit breaker triggered after %s consecutive errors", self.api_error_count)
            self.circuit_breaker = True
            self.scheduler.pause()
        logger.info("Cycle end at %s", datetime.now(tz=self.tz).isoformat())

    async def _process_symbol(self, symbol: str, cycle_start: datetime) -> None:
        raw = await asyncio.to_thread(self.data.fetch, symbol, self.cfg.timeframe, self.cfg.lookback_limit)
        df = compute_indicators(raw, self.params)
        df.index = df.index.tz_convert(self.tz)
        last_price = float(df["close"].iloc[-1])
        expected_close = _expected_candle_close(cycle_start, self.bar_minutes)
        last_close = df.index[-1]
        close_gap = abs((last_close - expected_close).total_seconds())
        tolerance = self.bar_minutes * 60
        candle_key = last_close.astimezone(timezone.utc).isoformat()
        if close_gap > tolerance:
            logger.warning(
                "Last candle (%s) outside tolerance vs expected close %s (gap=%ss)",
                last_close.isoformat(),
                expected_close.isoformat(),
                close_gap,
            )
        with self.state.candle_lock(symbol, candle_key) as acquired:
            if not acquired:
                logger.info("Skip %s at %s (already processed)", symbol, candle_key)
                return
            self.state.update_market_price(symbol, last_price)
            signal_result = compute_signal(df, self.params)
            parity_ok = self._parity_check(raw, df, signal_result.signal)
            account = self.broker.get_account()
            equity = _extract_equity(account, self.cfg.start_cash, self.cfg.quote_currency)
            position = self.broker.get_position(symbol)
            current_qty = float(position.get("qty", 0.0))
            target_fraction = max(self.cfg.min_size, min(self.cfg.max_pos_frac, self.cfg.risk_frac * signal_result.volatility_scale))
            leverage = self.cfg.max_leverage if self.cfg.max_leverage > 0 else 1.0
            target_notional = equity * target_fraction * leverage
            target_qty = target_notional / last_price if last_price > 0 else 0.0
            if signal_result.signal == Signal.SELL:
                target_qty = 0.0
            elif signal_result.signal == Signal.HOLD:
                target_qty = current_qty
            delta_qty = target_qty - current_qty
            min_qty = max(self.cfg.min_order_qty, 1e-8)
            order_info: Optional[Dict[str, object]] = None
            if signal_result.signal == Signal.BUY:
                side = "buy"
                order_qty = max(0.0, delta_qty)
            elif signal_result.signal == Signal.SELL:
                side = "sell"
                order_qty = max(0.0, -delta_qty)
            else:
                side = ""
                order_qty = 0.0

            if order_qty >= min_qty and signal_result.signal != Signal.HOLD:
                order_id = _make_order_id(symbol, candle_key, side, order_qty)
                try:
                    response = self.broker.place_order(
                        symbol,
                        side,
                        order_qty,
                        last_price,
                        order_type="market",
                        client_order_id=order_id,
                        timestamp=last_close.isoformat(),
                    )
                    order_info = response.info
                except BrokerError as exc:
                    logger.error("Order failure for %s: %s", symbol, exc)
            else:
                logger.info(
                    "No order for %s (signal=%s, delta_qty=%.8f)",
                    symbol,
                    signal_result.signal.value,
                    delta_qty,
                )
            self.state.record_cycle(
                symbol,
                candle_key,
                {
                    "signal": signal_result.signal.value,
                    "reason": signal_result.reason,
                    "volatility": signal_result.volatility,
                    "volatility_scale": signal_result.volatility_scale,
                    "target_fraction": target_fraction,
                    "equity": equity,
                    "price": last_price,
                    "order": order_info,
                    "parity_ok": parity_ok,
                },
            )

    def _parity_check(self, raw: List[List[float]], df: pd.DataFrame, latest_signal: Signal) -> bool:
        if not legacy_indicators or not legacy_last_signal:
            return True
        try:
            legacy_df = legacy_indicators(raw)
        except Exception as exc:  # pragma: no cover - legacy fetch failure
            logger.debug("Legacy indicator failure: %s", exc)
            return True
        legacy_signals: List[str] = []
        for idx in range(len(legacy_df)):
            window = legacy_df.iloc[: idx + 1]
            if len(window) < 2:
                continue
            sig = legacy_last_signal(window)
            legacy_signals.append(str(sig).upper())
        if not legacy_signals:
            return True
        ours = [res.signal.value for res in compute_signal_history(df, self.params, lookback=len(legacy_signals))]
        legacy_tail = legacy_signals[-len(ours) :]
        mismatches = sum(1 for a, b in zip(ours, legacy_tail) if a != b)
        if mismatches:
            logger.warning(
                "Backtest parity mismatch (%s diffs) legacy=%s ours=%s latest=%s",
                mismatches,
                legacy_tail,
                ours,
                latest_signal.value,
            )
            return False
        return True


async def main() -> None:
    bot = BotDaemon(settings)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
