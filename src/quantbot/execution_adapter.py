import ccxt
from quantbot.config import settings, get_symbol_params

try:
    from quantbot.position_sizing import compute_order_qty
except ImportError:  # fallback when run with src on sys.path
    from position_sizing import compute_order_qty


def _exchange_auth():
    if settings.exchange.lower() == "binance":
        ex = ccxt.binance(
            {
                "apiKey": settings.binance_key,
                "secret": settings.binance_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        return ex
    elif settings.exchange.lower() == "coinbase":
        ex = ccxt.coinbaseadvanced(
            {
                "apiKey": settings.coinbase_key,
                "secret": settings.coinbase_secret,
                "password": settings.coinbase_passphrase,
                "enableRateLimit": True,
            }
        )
        return ex
    else:
        raise ValueError("Unsupported exchange")


def place_order(side: str, symbol: str, qty: float):
    if settings.mode != "live":
        # Paper mode: mock fill at current price (execution to be improved)
        return {"status": "paper_fill", "symbol": symbol, "side": side, "qty": qty}
    ex = _exchange_auth()
    params = {"time_in_force": "GTC"}
    order = ex.create_order(
        symbol=symbol, type="market", side=side, amount=qty, params=params
    )
    return order


def place_signal(exchange, symbol, side, equity, last_price, realized_vol):
    # Spread guard
    spread_bps = None
    if settings.mode == "live":
        book = exchange.fetch_order_book(symbol, limit=10)
        bid, ask = book["bids"][0][0], book["asks"][0][0]
        spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 1e4
    else:
        # Assume a modest spread in paper mode if we can't get the book
        spread_bps = 5
    if spread_bps is not None and spread_bps > settings.spread_bps_limit:
        return None

    sym_params = get_symbol_params(symbol)
    target_vol = sym_params.get("target_vol", settings.vol_target)

    qty, px = compute_order_qty(
        exchange,
        symbol,
        equity,
        last_price,
        realized_vol,
        target_vol=target_vol,
        risk_frac=settings.risk_frac,
        max_pos_frac=settings.max_pos_frac,
    )
    if qty <= 0:
        return None

    params = {"time_in_force": "GTC", "postOnly": True}
    if settings.mode != "live":
        # Paper: return intention
        return {
            "status": "paper_signal",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": px or last_price,
            "spread_bps": spread_bps,
        }
    return exchange.create_order(symbol, "limit", side, qty, px or last_price, params)
