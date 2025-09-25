import argparse
from fastapi import FastAPI
from quantbot.config import settings, load_config_file
from quantbot.db import init_db, get_recent_trades

app = FastAPI(title="Crypto Momentum Bot Admin")


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/status")
def status():
    return {
        "mode": settings.mode,
        "exchange": settings.exchange,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
    }


@app.get("/trades")
async def trades(limit: int = 50):
    return await get_recent_trades(limit=limit)


@app.get("/metrics")
async def metrics():
    # TODO: compute and return PnL, winrate, drawdown from DB
    return {"pnl": None, "winrate": None, "max_drawdown": None}


def _cli_load_config():
    # For uvicorn run via `python -m quantbot.app --config path.yaml`
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path")
    args, _ = parser.parse_known_args()
    if getattr(args, "config_path", None):
        load_config_file(args.config_path)


_cli_load_config()
