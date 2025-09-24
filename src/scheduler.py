from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
from datetime import datetime
from .config import settings
from .data_adapter import fetch_ohlcv
from .signal_engine import compute_indicators, last_signal, volatility_target_size
from .execution_adapter import place_order, _exchange_auth, place_signal
from .db import log_trade
from .csv_logger import log_paper_trade


async def tick():
    ohlcv = fetch_ohlcv(settings.symbol, settings.timeframe, limit=300)
    df = compute_indicators(ohlcv)
    sig = last_signal(df)
    size_scale = volatility_target_size(df).iloc[-1]
    rationale = {"sig": sig, "size_scale": float(size_scale)}
    # Daily loss guard TODO: compute real PnL; placeholder False
    daily_loss_breached = False
    if daily_loss_breached:
        print("GUARD: daily loss limit breached; skipping")
        return

    last_close = float(df["close"].iloc[-1])
    # compute realized vol for sizing
    from .signal_engine import compute_realized_vol

    rv = compute_realized_vol(df)

    # In paper mode, avoid authenticated calls that require API keys
    ex = None
    if settings.mode == "live":
        ex = _exchange_auth()
        ex.load_markets()

    if sig in ("buy", "sell") and size_scale > 0:
        equity = 10000.0  # TODO: replace with actual equity tracking
        # Pass exchange only in live mode; in paper mode pass None to avoid API calls
        resp = place_signal(ex, settings.symbol, sig, equity, last_close, rv)
        qty = resp.get("qty") if isinstance(resp, dict) else None
        await log_trade(
            datetime.utcnow().isoformat(),
            settings.symbol,
            sig,
            float(qty or 0.0),
            float(last_close),
            0.0,
            str(rationale),
        )
        print("ORDER", resp)
        # CSV logging for paper mode
        if settings.mode != "live" and isinstance(resp, dict):
            log_paper_trade(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "exchange": settings.exchange,
                    "mode": settings.mode,
                    "symbol": settings.symbol,
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
        print("HOLD", rationale)


def start_scheduler():
    sched = AsyncIOScheduler(timezone="UTC")
    # run on the hour
    sched.add_job(tick, "cron", minute=0)
    sched.start()
    print("Scheduler started; running hourly.")


if __name__ == "__main__":
    asyncio.run(tick())
