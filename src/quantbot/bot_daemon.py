from __future__ import annotations

import asyncio
import hashlib
import json
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

from quantbot.broker import BrokerError, BrokerFactory
import argparse
from quantbot.config import settings, Settings, get_symbol_params, load_config_file
from quantbot.state import StateStore
from quantbot.smart_sizing import build_order_plan
from quantbot.strategy_macd import (
    Signal,
    SignalResult,
    StrategyParams,
    apply_cooldown,
    compute_indicators,
    compute_signal,
    compute_signal_history,
)

try:  # optional legacy comparison
    from quantbot.signal_engine import compute_indicators as legacy_indicators
    from quantbot.signal_engine import last_signal as legacy_last_signal
except Exception:  # pragma: no cover - optional dependency
    legacy_indicators = None
    legacy_last_signal = None


logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, cfg: Settings) -> None:
        provider = (cfg.data_provider or cfg.exchange or "binance").lower()
        if provider == "binance":
            self.exchange = ccxt.binance(
                {"enableRateLimit": True, "options": {"defaultType": "spot"}}
            )
        elif provider in {"coinbase", "coinbaseadvanced"}:
            self.exchange = ccxt.coinbaseadvanced({"enableRateLimit": True})
        else:
            raise ValueError(f"Unsupported data provider: {provider}")
        self.cfg = cfg
        self.markets_cache = {}  # Cache for market info (min_market_funds, etc.)

        # Load markets for minimum sizing info (safe for both live and paper mode)
        try:
            self.exchange.load_markets()
            self._cache_market_info()
        except Exception as exc:
            logger.warning(
                "load_markets failed (%s); proceeding without cached markets", exc
            )

    def _cache_market_info(self) -> None:
        """Cache exchange minimum requirements for each symbol."""
        try:
            for symbol in self.cfg.symbols:
                ccxt_symbol = self.cfg.as_ccxt_symbol(symbol)
                if ccxt_symbol in self.exchange.markets:
                    market = self.exchange.markets[ccxt_symbol]
                    limits = market.get("limits", {})

                    # Extract market minimums
                    cost_min = None
                    amount_min = None

                    if "cost" in limits and limits["cost"].get("min") is not None:
                        cost_min = float(limits["cost"]["min"])
                    if "amount" in limits and limits["amount"].get("min") is not None:
                        amount_min = float(limits["amount"]["min"])

                    # For lot step, get precision info
                    lot_step = None
                    if "precision" in market and "amount" in market["precision"]:
                        # Convert precision to step size (e.g., 5 decimals = 0.00001 step)
                        amount_precision = market["precision"]["amount"]
                        if isinstance(amount_precision, int) and amount_precision > 0:
                            lot_step = 10 ** (-amount_precision)

                    self.markets_cache[symbol] = {
                        "min_market_funds": cost_min,  # minimum notional USD
                        "min_qty": amount_min,  # minimum base asset quantity
                        "lot_step": lot_step,  # quantity step size
                        "ccxt_symbol": ccxt_symbol,
                    }

                    logger.info(
                        f"Cached market info for {symbol}: min_funds=${cost_min}, "
                        f"min_qty={amount_min}, lot_step={lot_step}"
                    )
        except Exception as exc:
            logger.warning(f"Failed to cache market info: {exc}")

    def get_market_info(self, symbol: str) -> Dict:
        """Get cached market info for a symbol."""
        return self.markets_cache.get(symbol, {})

    def fetch(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        market_symbol = self.cfg.as_ccxt_symbol(symbol)
        return self.exchange.fetch_ohlcv(
            market_symbol, timeframe=timeframe, limit=limit
        )


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


def _extract_equity(
    account: Dict[str, object], default: float, quote_currency: str
) -> float:
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
        # Strategy overrides are loaded from YAML strategy_overrides only
        self.maker_offset = float(getattr(cfg, "maker_offset_bps", 1.0)) / 10_000.0
        self.params = StrategyParams(
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
            macd_signal=cfg.macd_signal,
            adx_length=cfg.adx_len,
            adx_threshold=cfg.adx_threshold,
            macd_cross_grace_bars=cfg.macd_cross_grace_bars,
            macd_thrust_bars=cfg.macd_thrust_bars,
            vol_lookback=cfg.vol_lookback,
            target_daily_vol=cfg.vol_target,
            min_fraction=cfg.min_size,
            max_fraction=cfg.max_size,
            cooldown_bars=int(cfg.cooldown_bars),
            entry_order_type=cfg.entry_order_type,
            exit_order_type=cfg.exit_order_type,
            bar_minutes=self.bar_minutes,
        )
        self.risk_frac = cfg.risk_frac
        self.entry_order_type = self.params.entry_order_type
        self.exit_order_type = self.params.exit_order_type
        overrides = self._load_strategy_overrides(cfg.symbol)
        if overrides:
            self._apply_strategy_overrides(overrides)

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").replace("-", "").replace("_", "").upper()

    def _load_strategy_overrides(self, symbol: str) -> Dict[str, object]:
        """Return per-symbol overrides from YAML via config.get_symbol_params."""
        try:
            return dict(get_symbol_params(symbol))
        except Exception:
            return {}

    def _apply_strategy_overrides(self, overrides: Dict[str, object]) -> None:
        if not overrides:
            return
        params_dict = {**self.params.__dict__}
        mapping = {
            "fast": "macd_fast",
            "slow": "macd_slow",
            "signal": "macd_signal",
            "adx_len": "adx_length",
            "adx_th": "adx_threshold",
            "macd_cross_grace_bars": "macd_cross_grace_bars",
            "grace_bars": "macd_cross_grace_bars",
            "grace_window": "macd_cross_grace_bars",
            "macd_thrust_bars": "macd_thrust_bars",
            "thrust_bars": "macd_thrust_bars",
            "vol_lb": "vol_lookback",
            "daily_vol_target": "target_daily_vol",
            "cooldown_bars": "cooldown_bars",
            "entry_order_type": "entry_order_type",
            "exit_order_type": "exit_order_type",
        }
        for key, attr in mapping.items():
            if key in overrides and overrides[key] is not None:
                value = overrides[key]
                if attr in {
                    "macd_fast",
                    "macd_slow",
                    "macd_signal",
                    "adx_length",
                    "vol_lookback",
                    "cooldown_bars",
                    "macd_cross_grace_bars",
                    "macd_thrust_bars",
                }:
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                elif attr == "target_daily_vol":
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                params_dict[attr] = value
        params_dict["bar_minutes"] = self.bar_minutes
        params_dict["vol_lookback"] = int(
            params_dict.get("vol_lookback", self.params.vol_lookback)
        )
        params_dict["cooldown_bars"] = int(
            params_dict.get("cooldown_bars", self.params.cooldown_bars)
        )
        self.params = StrategyParams(**params_dict)
        self.entry_order_type = self.params.entry_order_type
        self.exit_order_type = self.params.exit_order_type
        if "risk_frac" in overrides and overrides["risk_frac"] is not None:
            try:
                self.risk_frac = float(overrides["risk_frac"])
            except (TypeError, ValueError):
                pass
        if (
            "maker_offset_bps" in overrides
            and overrides["maker_offset_bps"] is not None
        ):
            try:
                self.maker_offset = float(overrides["maker_offset_bps"]) / 10_000.0
            except (TypeError, ValueError):
                pass

    def _cron_trigger(self) -> CronTrigger:
        if self.bar_minutes >= 60:
            return CronTrigger(minute=0, second=5, timezone=self.tz)
        if self.bar_minutes == 30:
            return CronTrigger(minute="0,30", second=5, timezone=self.tz)
        if self.bar_minutes == 15:
            return CronTrigger(minute="0,15,30,45", second=5, timezone=self.tz)
        return CronTrigger(second=5, timezone=self.tz)

    def _maker_price(
        self, reference_price: float, side: str, offset: Optional[float] = None
    ) -> float:
        price = float(reference_price)
        bias = self.maker_offset if offset is None else float(offset)
        bias = max(0.0, bias)
        if price <= 0 or bias == 0.0:
            return max(price, 0.0)
        if side.lower() == "buy":
            return max(0.0, price * (1 - bias))
        return max(0.0, price * (1 + bias))

    def _order_plan(
        self, side: str, qty: float, last_price: float, reference_price: Optional[float]
    ) -> Dict[str, object]:
        base_price = (
            reference_price if reference_price and reference_price > 0 else last_price
        )
        preference = (
            self.entry_order_type if side.lower() == "buy" else self.exit_order_type
        )
        pref = (preference or "market").lower()
        plan = {
            "order_type": "market",
            "time_in_force": "GTC",
            "post_only": False,
            "reduce_only": side.lower() == "sell",
            "price": last_price,
        }
        if pref in {"limit", "limit_post_only"}:
            plan["order_type"] = "limit"
            plan["post_only"] = pref == "limit_post_only"
            plan["price"] = self._maker_price(base_price, side)
        elif pref == "market":
            plan["order_type"] = "market"
            plan["price"] = last_price
            plan["post_only"] = False
        else:
            plan["order_type"] = pref
            plan["price"] = base_price
            plan["post_only"] = False
        return plan

    def _place_order_with_retry(
        self,
        symbol: str,
        side: str,
        qty: float,
        plan: Dict[str, object],
        order_id: str,
        timestamp: str,
        last_price: float,
    ):
        attempts = 0
        price = float(plan.get("price", last_price))
        while True:
            try:
                return self.broker.place_order(
                    symbol,
                    side,
                    qty,
                    price,
                    order_type=str(plan.get("order_type", "market")),
                    time_in_force=str(plan.get("time_in_force", "GTC")),
                    post_only=bool(plan.get("post_only", False)),
                    reduce_only=bool(plan.get("reduce_only", False)),
                    client_order_id=order_id,
                    timestamp=timestamp,
                )
            except BrokerError as exc:
                if (
                    plan.get("order_type") == "limit"
                    and plan.get("post_only")
                    and attempts < 1
                ):
                    attempts += 1
                    price = self._maker_price(
                        last_price, side, self.maker_offset * (attempts + 1)
                    )
                    plan["price"] = price
                    logger.warning(
                        "Retrying post-only limit %s with wider offset (attempt %s)",
                        order_id,
                        attempts,
                    )
                    continue
                raise

    def _last_exit_timestamp(self, symbol: str) -> Optional[datetime]:
        record = self.state.get_last_action(symbol)
        if not record:
            return None
        meta = record.get("meta") or {}
        ts_str = meta.get("last_exit_ts") or record.get("ts")
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return None

    async def start(self) -> None:
        _setup_logging(self.cfg)
        self._setup_signal_handlers()
        trigger = self._cron_trigger()
        self.scheduler.add_job(
            self.run_cycle,
            trigger=trigger,
            max_instances=self.cfg.max_instances,
            coalesce=True,
            misfire_grace_time=self.cfg.misfire_grace_time,
        )
        self.scheduler.add_listener(
            self._job_listener, EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        self.scheduler.start()
        logger.info(
            "Bot daemon started in %s mode (dry_run=%s) for symbols %s",
            self.cfg.env,
            self.cfg.dry_run,
            ", ".join(self.cfg.symbol_universe),
        )
        product_map = {
            sym: sym.replace("/", "-") for sym in self.cfg.symbol_universe
        }
        if product_map:
            logger.info("PRODUCT_MAP %s", product_map)
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
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self._shutdown(s))
                )
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
            logger.error(
                "Circuit breaker triggered after %s consecutive errors",
                self.api_error_count,
            )
            self.circuit_breaker = True
            self.scheduler.pause()
        logger.info("Cycle end at %s", datetime.now(tz=self.tz).isoformat())

    async def _process_symbol(self, symbol: str, cycle_start: datetime) -> None:
        raw = await asyncio.to_thread(
            self.data.fetch, symbol, self.cfg.timeframe, self.cfg.lookback_limit
        )
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
            position = self.broker.get_position(symbol)
            current_qty = float(position.get("qty", 0.0))
            position_state = "FLAT"
            if current_qty > 0:
                position_state = "LONG"
            elif current_qty < 0:
                position_state = "SHORT"

            signal_result = compute_signal(
                df, self.params, position_state=position_state
            )
            cooldown_ts = self._last_exit_timestamp(symbol)
            current_ts = (
                last_close.to_pydatetime()
                if hasattr(last_close, "to_pydatetime")
                else datetime.fromisoformat(str(last_close))
            )
            signal_result = apply_cooldown(
                signal_result,
                self.params,
                last_exit_ts=cooldown_ts,
                current_ts=current_ts,
            )
            if signal_result.meta.get("cooldown_active"):
                logger.debug(
                    "Cooldown active for %s (bars_since_exit=%s of %s)",
                    symbol,
                    signal_result.meta.get("bars_since_exit"),
                    signal_result.meta.get("cooldown_bars"),
                )
            parity_ok = self._parity_check(raw, df, signal_result.signal)
            account = self.broker.get_account()
            equity = _extract_equity(
                account, self.cfg.start_cash, self.cfg.quote_currency
            )
            target_fraction = max(
                self.cfg.min_size,
                min(
                    self.cfg.max_pos_frac,
                    self.risk_frac * signal_result.volatility_scale,
                ),
            )

            # Get exchange-specific market info for smart sizing
            market_info = self.data.get_market_info(symbol)

            # Use smart sizing with dynamic exchange minimums
            order_plan = build_order_plan(
                signal=signal_result.signal.value,
                symbol=symbol,
                equity_usd=equity,
                target_fraction=target_fraction,
                price=last_price,
                lot_step=market_info.get("lot_step"),
                min_qty=market_info.get("min_qty"),
                exch_min_notional=market_info.get("min_market_funds"),
            )

            order_info: Optional[Dict[str, object]] = None
            delta_qty = 0.0

            if order_plan and signal_result.signal != Signal.HOLD:
                order_id = _make_order_id(
                    symbol, candle_key, order_plan["side"], order_plan["qty"]
                )
                try:
                    response = self._place_order_with_retry(
                        symbol,
                        order_plan["side"],
                        order_plan["qty"],
                        order_plan,
                        order_id,
                        last_close.isoformat(),
                        last_price,
                    )
                    order_info = response.info
                    action_meta = dict(order_plan)
                    if isinstance(order_info, dict):
                        action_meta.update(order_info)
                    action_meta["qty"] = order_plan["qty"]
                    if order_plan["side"].lower() == "sell":
                        action_meta.setdefault("last_exit_ts", last_close.isoformat())
                    self.state.set_last_action(
                        symbol,
                        order_plan["side"].upper(),
                        ts=last_close.isoformat(),
                        order_id=response.order_id,
                        meta=action_meta,
                    )
                    # Calculate delta_qty for logging
                    if order_plan["side"].lower() == "buy":
                        delta_qty = order_plan["qty"]
                    else:
                        delta_qty = -order_plan["qty"]
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
                    "order_plan": order_plan,
                    "signal_meta": signal_result.meta,
                    "parity_ok": parity_ok,
                    "cooldown_bars": self.params.cooldown_bars,
                    "maker_offset": self.maker_offset,
                },
            )

    def _parity_check(
        self, raw: List[List[float]], df: pd.DataFrame, latest_signal: Signal
    ) -> bool:
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
        ours = [
            res.signal.value
            for res in compute_signal_history(
                df, self.params, lookback=len(legacy_signals)
            )
        ]
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
    parser = argparse.ArgumentParser(description="Run trading bot daemon")
    parser.add_argument(
        "--config", dest="config_path", help="Path to YAML run config", default=None
    )
    args, _ = parser.parse_known_args()
    cfg = settings
    if args.config_path:
        cfg = load_config_file(args.config_path)
    bot = BotDaemon(cfg)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
