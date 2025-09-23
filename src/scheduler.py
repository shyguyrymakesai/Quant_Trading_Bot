from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
from datetime import datetime
from .config import settings
from .data_adapter import fetch_ohlcv
from .signal_engine import compute_indicators, last_signal, volatility_target_size
from .execution_adapter import place_order
from .db import log_trade

async def tick():
    ohlcv = fetch_ohlcv(settings.symbol, settings.timeframe, limit=300)
    df = compute_indicators(ohlcv)
    sig = last_signal(df)
    size_scale = volatility_target_size(df).iloc[-1]
    rationale = {"sig": sig, "adx_thr": settings.adx_threshold, "size_scale": float(size_scale)}
    if sig in ("buy","sell") and size_scale > 0:
        # naive qty calculation: notional in base currency / last close
        last_close = df['close'].iloc[-1]
        notional = min(settings.max_position_notional, 25.0)  # hard cap
        qty = round((notional * size_scale) / last_close, 6)
        resp = place_order(sig, settings.symbol, qty)
        await log_trade(datetime.utcnow().isoformat(), settings.symbol, sig, qty, float(last_close), 0.0, str(rationale))
        print("TRADE", resp)
    else:
        print("HOLD", rationale)

def start_scheduler():
    sched = AsyncIOScheduler(timezone="UTC")
    # run on the hour
    sched.add_job(tick, "cron", minute=0)
    sched.start()
    print("Scheduler started; running hourly.")

if __name__ == "__main__":
    asyncio.run(tick())
