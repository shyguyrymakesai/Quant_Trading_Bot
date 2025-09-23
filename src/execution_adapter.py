import ccxt
from .config import settings

def _exchange_auth():
    if settings.exchange.lower() == "binance":
        ex = ccxt.binance({
            "apiKey": settings.binance_key,
            "secret": settings.binance_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        return ex
    elif settings.exchange.lower() == "coinbase":
        ex = ccxt.coinbaseadvanced({
            "apiKey": settings.coinbase_key,
            "secret": settings.coinbase_secret,
            "password": settings.coinbase_passphrase,
            "enableRateLimit": True,
        })
        return ex
    else:
        raise ValueError("Unsupported exchange")

def place_order(side: str, symbol: str, qty: float):
    if settings.mode != "live":
        # Paper mode: mock fill at current price (execution to be improved)
        return {"status": "paper_fill", "symbol": symbol, "side": side, "qty": qty}
    ex = _exchange_auth()
    params = {}
    order = ex.create_order(symbol=symbol, type="market", side=side, amount=qty, params=params)
    return order
