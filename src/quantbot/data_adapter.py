import ccxt
import asyncio
from quantbot.config import settings

def _exchange():
    if settings.exchange.lower() == "binance":
        ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        # NOTE: For testnet, user must set base URLs via env or ccxt options
        return ex
    elif settings.exchange.lower() == "coinbase":
        ex = ccxt.coinbaseadvanced({
            "enableRateLimit": True,
        })
        return ex
    else:
        raise ValueError("Unsupported exchange")

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000):
    ex = _exchange()
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
