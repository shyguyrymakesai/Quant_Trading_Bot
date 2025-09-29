from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
import json
import boto3
from datetime import datetime
from quantbot.config import settings
from quantbot.data_adapter import fetch_ohlcv
from quantbot.signal_engine import (
    compute_indicators,
    last_signal,
    volatility_target_size,
)
from quantbot.execution_adapter import place_order, _exchange_auth, place_signal
from quantbot.db import log_trade
from quantbot.csv_logger import log_paper_trade


async def log_to_s3(data: dict, bucket: str = "quant-bot-trades-969932165253"):
    """Log detailed data to S3 bucket"""
    try:
        s3 = boto3.client('s3')
        timestamp = datetime.utcnow()
        key = f"logs/{timestamp.strftime('%Y/%m/%d')}/{timestamp.strftime('%H-%M-%S')}-{data.get('symbol', 'unknown').replace('/', '-')}.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )
        print(f"Logged to S3: s3://{bucket}/{key}")
    except Exception as e:
        print(f"Failed to log to S3: {e}")


async def process_symbol(symbol: str):
    """Process a single symbol and return detailed results"""
    print(f"\n=== Processing {symbol} ===")
    
    try:
        # Fetch data
        ohlcv = fetch_ohlcv(symbol, settings.timeframe, limit=300)
        df = compute_indicators(ohlcv)
        sig = last_signal(df)
        size_scale = volatility_target_size(df).iloc[-1]
        rationale = {"sig": sig, "size_scale": float(size_scale)}
        
        # Get latest market data
        last_close = float(df["close"].iloc[-1])
        last_high = float(df["high"].iloc[-1])
        last_low = float(df["low"].iloc[-1])
        last_volume = float(df["volume"].iloc[-1])
        
        # Get technical indicators for logging
        macd_line = float(df["macd"].iloc[-1]) if "macd" in df.columns else None
        macd_signal = float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else None
        macd_hist = float(df["macd_hist"].iloc[-1]) if "macd_hist" in df.columns else None
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else None
        
        # Compute realized vol
        from quantbot.signal_engine import compute_realized_vol
        rv = compute_realized_vol(df)
        
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
                "realized_vol": float(rv) if rv is not None else None,
            },
            "rationale": rationale,
        }
        
        # In paper mode, avoid authenticated calls
        ex = None
        if settings.mode == "live":
            ex = _exchange_auth()
            ex.load_markets()
        
        if sig in ("buy", "sell") and size_scale > 0:
            equity = 10000.0  # TODO: replace with actual equity tracking
            resp = place_signal(ex, symbol, sig, equity, last_close, rv)
            qty = resp.get("qty") if isinstance(resp, dict) else None
            
            # Add order details to log
            log_data["order"] = {
                "action": "ORDER_PLACED",
                "side": sig,
                "quantity": float(qty or 0.0),
                "price": float(resp.get("price") or last_close),
                "response": resp
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
                log_paper_trade({
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
                })
        else:
            log_data["order"] = {
                "action": "HOLD",
                "reason": f"Signal: {sig}, Size Scale: {size_scale}"
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
            "error_type": type(e).__name__
        }
        print(f"ERROR processing {symbol}: {e}")
        await log_to_s3(error_log)
        return error_log


async def tick():
    """Main tick function - processes all configured symbols"""
    print(f"\n🤖 Bot run started at {datetime.utcnow().isoformat()}")
    print(f"Mode: {settings.mode} | Exchange: {settings.exchange} | Timeframe: {settings.timeframe}")
    
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
        "results": results
    }
    
    # Log summary to S3
    await log_to_s3(summary, bucket=settings.s3_bucket_trades if hasattr(settings, 's3_bucket_trades') else "quant-bot-trades-969932165253")
    
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
