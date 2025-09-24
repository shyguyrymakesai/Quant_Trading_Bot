import math


def floor_to_step(x, step):
    if step is None or step == 0:
        return x
    return math.floor(x / step) * step


def round_price(p, tick_size):
    if tick_size is None or tick_size == 0:
        return p
    return round(p / tick_size) * tick_size


def compute_order_qty(
    exchange,
    symbol,
    equity,
    last_price,
    realized_vol,
    target_vol=0.02,
    risk_frac=0.10,
    max_pos_frac=0.30,
):
    """
    Coinbase Advanced supports fractional sizes; we must respect exchange limits.
    Returns (qty, px) where qty is rounded to step, px rounded to tick (for limit orders).
    """
    amount_step = None
    min_amount = None
    min_cost = None
    tick_size = None
    if exchange is not None:
        try:
            m = exchange.market(symbol)
            amount_step = m.get("precision", {}).get("amount") or m.get(
                "limits", {}
            ).get("amount", {}).get("step")
            min_amount = m.get("limits", {}).get("amount", {}).get("min")
            min_cost = m.get("limits", {}).get("cost", {}).get("min")
            tick_size = m.get("precision", {}).get("price") or m.get("limits", {}).get(
                "price", {}
            ).get("min")
        except Exception:
            # In paper mode or when markets aren't loaded, proceed with sane defaults
            amount_step = None
            min_amount = None
            min_cost = None
            tick_size = None

    if realized_vol is None or realized_vol <= 0:
        return 0.0, None

    vol_scale = min(1.0, max(0.0, target_vol / realized_vol))
    target_notional = float(equity) * float(risk_frac) * float(vol_scale)
    max_pos_notional = float(equity) * float(max_pos_frac)
    target_notional = min(target_notional, max_pos_notional)

    if min_cost:
        target_notional = max(target_notional, float(min_cost) * 1.05)

    qty_raw = target_notional / max(float(last_price), 1e-9)
    qty = floor_to_step(qty_raw, amount_step)

    if min_amount:
        qty = max(qty, float(min_amount))

    if qty <= 0:
        return 0.0, None

    px = round_price(float(last_price), tick_size)
    return qty, px
