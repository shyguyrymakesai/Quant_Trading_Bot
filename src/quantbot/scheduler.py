import asyncio
import json
from typing import Optional

import boto3
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from quantbot.config import settings, get_symbol_params
from quantbot.data_adapter import fetch_ohlcv
from quantbot.strategy_macd import (
    StrategyParams,
    compute_indicators as compute_strategy_indicators,
    compute_signal as compute_strategy_signal,
)
from quantbot.execution_adapter import _exchange_auth, place_signal
from quantbot.db import log_trade
from quantbot.csv_logger import log_paper_trade


async def log_to_s3(data: dict, bucket: str = "quant-bot-trades-969932165253"):
    """Log detailed data to S3 bucket"""
    try:
        s3 = boto3.client("s3")
        timestamp = datetime.utcnow()
        key = f"logs/{timestamp.strftime('%Y/%m/%d')}/{timestamp.strftime('%H-%M-%S')}-{data.get('symbol', 'unknown').replace('/', '-')}.json"

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json",
        )
        print(f"Logged to S3: s3://{bucket}/{key}")
    except Exception as e:
        print(f"Failed to log to S3: {e}")


def _timeframe_to_minutes(tf: str) -> int:
    tf = (tf or "").strip().lower()
    if tf.endswith("m"):
        return max(1, int(float(tf[:-1] or 0)))
    if tf.endswith("h"):
        return max(1, int(float(tf[:-1] or 0) * 60))
    if tf.endswith("d"):
        return max(1, int(float(tf[:-1] or 0) * 1440))
    if tf.isdigit():
        return max(1, int(tf))
    return 60


def _safe_latest(df: pd.DataFrame, column: str) -> Optional[float]:
    if df is None or column not in df.columns:
        return None
    series = df[column]
    if series is None:
        return None
    try:
        if hasattr(series, "dropna"):
            series = series.dropna()
        if len(series) == 0:
            return None
        value = series.iloc[-1]
    except (IndexError, KeyError, AttributeError):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_strategy_params(symbol: str) -> StrategyParams:
    base_kwargs = {
        "macd_fast": settings.macd_fast,
        "macd_slow": settings.macd_slow,
        "macd_signal": settings.macd_signal,
        "adx_length": settings.adx_len,
        "adx_threshold": settings.adx_threshold,
        "macd_cross_grace_bars": getattr(settings, "macd_cross_grace_bars", 3),
        "macd_thrust_bars": getattr(settings, "macd_thrust_bars", 2),
        "vol_lookback": settings.vol_lookback,
        "target_daily_vol": settings.vol_target,
        "min_fraction": settings.min_size,
        "max_fraction": settings.max_size,
        "cooldown_bars": int(getattr(settings, "cooldown_bars", 0)),
        "entry_order_type": settings.entry_order_type,
        "exit_order_type": settings.exit_order_type,
        "bar_minutes": _timeframe_to_minutes(settings.timeframe),
    }
    overrides = get_symbol_params(symbol)
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
            elif attr in {"target_daily_vol", "min_fraction", "max_fraction"}:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
            base_kwargs[attr] = value
    return StrategyParams(**base_kwargs)


async def process_symbol(symbol: str):
    """Process a single symbol and return detailed results"""
    print(f"\n=== Processing {symbol} ===")

    try:
        # Fetch data
        ohlcv = fetch_ohlcv(symbol, settings.timeframe, limit=300)
        params = _build_strategy_params(symbol)
        df = compute_strategy_indicators(ohlcv, params)
        signal_result = compute_strategy_signal(df, params, position_state="FLAT")
        sig_enum = signal_result.signal
        sig = sig_enum.value.lower()
        size_scale = float(signal_result.volatility_scale)
        rationale = {
            "sig": sig,
            "size_scale": float(size_scale),
            "reason": signal_result.reason,
        }
        diag = dict(signal_result.meta or {})

        # Get latest market data
        last_close = _safe_latest(df, "close")
        last_high = _safe_latest(df, "high")
        last_low = _safe_latest(df, "low")
        last_volume = _safe_latest(df, "volume")

        # Get technical indicators for logging
        macd_line = _safe_latest(df, "macd")
        macd_signal = _safe_latest(df, "macd_signal")
        macd_hist = _safe_latest(df, "macd_hist")
        adx = _safe_latest(df, "adx")

        # Compute realized vol
        rv = float(signal_result.volatility)

        # Daily loss guard
        daily_loss_breached = False
        if daily_loss_breached:
            print(f"GUARD: daily loss limit breached for {symbol}; skipping")
            return

        # Prepare detailed log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "exchange": settings.exchange,
            "mode": settings.mode,
            "timeframe": settings.timeframe,
            "signal": sig,
            "size_scale": float(size_scale),
            "market_data": {
                "close": last_close,
                "high": last_high,
                "low": last_low,
                "volume": last_volume,
            },
            "indicators": {
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_hist,
                "adx": adx,
                "realized_vol": rv,
                "trend_ok": bool(diag.get("trend_ok")),
                "cross_up_now": bool(diag.get("cross_up_now")),
                "cross_up_within_k": bool(diag.get("cross_up_within_k")),
                "thrust_up": bool(diag.get("thrust_up")),
                "cooldown_ok": bool(diag.get("cooldown_ok", True)),
                "position": diag.get("position") or diag.get("position_state"),
                "cross_down_now": bool(diag.get("cross_down_now")),
                "cross_down_within_k": bool(diag.get("cross_down_within_k")),
                "thrust_down": bool(diag.get("thrust_down")),
            },
            "rationale": rationale,
        }

        # In paper mode, avoid authenticated calls
        ex = None
        if settings.mode == "live":
            ex = _exchange_auth()
            ex.load_markets()

        can_trade = sig in ("buy", "sell") and size_scale > 0 and last_close is not None

        if can_trade:
            equity = 10000.0  # TODO: replace with actual equity tracking
            resp = place_signal(ex, symbol, sig, equity, last_close, rv)
            qty = resp.get("qty") if isinstance(resp, dict) else None

            # Add order details to log
            log_data["order"] = {
                "action": "ORDER_PLACED",
                "side": sig,
                "quantity": float(qty or 0.0),
                "price": float(resp.get("price") or last_close),
                "response": resp,
                "reason": signal_result.reason,
            }

            await log_trade(
                datetime.utcnow().isoformat(),
                symbol,
                sig,
                float(qty or 0.0),
                float(last_close),
                0.0,
                str(rationale),
            )
            print(f"ORDER {symbol}:", resp)

            # CSV logging for paper mode
            if settings.mode != "live" and isinstance(resp, dict):
                log_paper_trade(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "exchange": settings.exchange,
                        "mode": settings.mode,
                        "symbol": symbol,
                        "side": sig,
                        "qty": float(qty or 0.0),
                        "price": float(resp.get("price") or last_close),
                        "last_close": float(last_close),
                        "realized_vol": float(rv or 0.0),
                        "size_scale": float(size_scale),
                        "spread_bps": float(resp.get("spread_bps") or 0.0),
                        "status": resp.get("status"),
                        "rationale": str(rationale),
                    }
                )
        else:
            if last_close is None and sig in ("buy", "sell"):
                rationale["reason"] = f"{signal_result.reason}|missing_market_data"
                sig = "hold"
                log_data["signal"] = sig
            log_data["order"] = {
                "action": "HOLD",
                "reason": rationale["reason"],
            }
            print(f"HOLD {symbol}:", rationale)

        # Log to S3
        await log_to_s3(log_data)

        return log_data

    except Exception as e:
        error_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        print(f"ERROR processing {symbol}: {e}")
        await log_to_s3(error_log)
        return error_log


async def tick():
    """Main tick function - processes all configured symbols"""
    print(f"\n🤖 Bot run started at {datetime.utcnow().isoformat()}")
    print(
        f"Mode: {settings.mode} | Exchange: {settings.exchange} | Timeframe: {settings.timeframe}"
    )

    # Get symbols to process
    symbols_to_process = []

    # Use symbols from environment variable if available
    if settings.symbols and len(settings.symbols) > 0:
        symbols_to_process = settings.symbols
        print(f"Processing symbols from environment: {symbols_to_process}")
    else:
        # Fallback to single symbol from config
        symbols_to_process = [settings.symbol]
        print(f"Processing single symbol from config: {symbols_to_process}")

    # Convert Coinbase format (BTC-USD) to standard format (BTC/USD) if needed
    normalized_symbols = []
    for symbol in symbols_to_process:
        if "-" in symbol and "/" not in symbol:
            normalized_symbol = symbol.replace("-", "/")
            normalized_symbols.append(normalized_symbol)
            print(f"Normalized {symbol} -> {normalized_symbol}")
        else:
            normalized_symbols.append(symbol)

    # Process each symbol
    results = []
    for symbol in normalized_symbols:
        result = await process_symbol(symbol)
        results.append(result)

    # Create summary log
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_summary": {
            "total_symbols": len(normalized_symbols),
            "symbols_processed": normalized_symbols,
            "mode": settings.mode,
            "exchange": settings.exchange,
        },
        "results": results,
    }
    summary["symbol"] = "SUMMARY"

    # Log summary to S3
    await log_to_s3(
        summary,
        bucket=(
            settings.s3_bucket_trades
            if hasattr(settings, "s3_bucket_trades")
            else "quant-bot-trades-969932165253"
        ),
    )

    print(f"\n✅ Bot run completed. Processed {len(normalized_symbols)} symbols.")
    print("Bot run completed.")


def start_scheduler():
    sched = AsyncIOScheduler(timezone="UTC")
    # run twice per hour
    sched.add_job(tick, "cron", minute="0,30")
    sched.start()
    print("Scheduler started; running every 30 minutes.")


if __name__ == "__main__":
    asyncio.run(tick())
