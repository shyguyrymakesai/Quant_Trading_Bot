"""
Smart position sizing with dynamic exchange minimums.

Priority order:
1. Uses volatility targeting (equity * target_fraction)
2. If below exchange min_market_funds, bumps to exchange requirement
3. If below human preference floor, uses FLAT_ORDER_NOTIONAL to avoid dust orders
4. Enforces exchange lot_step rounding and maker-only pricing

Fetches actual exchange minimums dynamically:
- BTC-USD typically ~$1 min_market_funds
- ETH-USD typically ~$1 min_market_funds
- Smaller alts vary ($0.10 to $5+)
"""

import os
from decimal import Decimal, ROUND_DOWN
from typing import Tuple, Optional

# Environment configuration
MIN_NOTIONAL_FLOOR = Decimal(
    os.getenv("MIN_NOTIONAL_FLOOR", "1")
)  # Human floor to avoid dust (respects exchange min)
FLAT_ORDER_NOTIONAL = Decimal(os.getenv("FLAT_ORDER_NOTIONAL", "1"))
MAKER_ONLY = os.getenv("MAKER_ONLY", "true").lower() == "true"
MAKER_OFFSET = Decimal(os.getenv("MAKER_OFFSET", "0.00015"))


def _round_step(qty: Decimal, step: Decimal | None) -> Decimal:
    """Round quantity DOWN to exchange step size."""
    if not step or step == 0:
        return qty
    # round DOWN to exchange step size
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def plan_size(
    equity_usd: float,
    target_fraction: float,
    price: float,
    *,
    lot_step: float | None = None,  # exchange base-asset increment (e.g., 0.00001 BTC)
    min_qty: float | None = None,  # exchange minimum base qty
    exch_min_notional: float | None = None  # exchange minimum notional (USD)
) -> Tuple[Decimal, Decimal, str]:
    """
    Returns (qty_base, notional_usd, sizing_reason)

    Logic:
    1. Calculate vol targeting notional (equity * target_fraction)
    2. If exchange min_market_funds > vol_notional, use exchange minimum
    3. If vol_notional < human floor (MIN_NOTIONAL_FLOOR), use FLAT_ORDER_NOTIONAL
    4. Enforce exchange min qty/notional and round to lot_step
    """
    price = Decimal(str(price))
    equity = Decimal(str(equity_usd))
    tf = Decimal(str(target_fraction))

    vol_notional = equity * tf  # your usual vol targeting notional
    human_floor = MIN_NOTIONAL_FLOOR  # human preference to avoid $1 dust orders
    chosen_notional = vol_notional
    reason = "vol_target"

    # First, check exchange minimum (always respect exchange limits)
    if exch_min_notional:
        emn = Decimal(str(exch_min_notional))
        if vol_notional < emn:
            chosen_notional = emn
            reason = "exchange_min_notional"

    # Then apply human floor if still below our preference (but above exchange min)
    if chosen_notional < human_floor and reason != "exchange_min_notional":
        chosen_notional = FLAT_ORDER_NOTIONAL
        reason = "human_min_floor"

    qty = chosen_notional / price

    # Round to step & enforce min qty
    qty = _round_step(qty, Decimal(str(lot_step)) if lot_step else None)
    if min_qty and qty < Decimal(str(min_qty)):
        # bump to min_qty if possible
        qty = Decimal(str(min_qty))
        chosen_notional = qty * price
        reason = "exchange_min_qty"

    # If rounding zeroed it out, return zeros (caller can skip)
    if qty <= 0:
        return Decimal("0"), Decimal("0"), "too_small_after_rounding"

    return qty, chosen_notional, reason


def maker_limit_price(side: str, mid_price: float) -> float:
    """
    Calculate maker-only limit price with offset.
    For buy orders: bid below mid by MAKER_OFFSET
    For sell orders: ask above mid by MAKER_OFFSET
    """
    p = Decimal(str(mid_price))
    off = MAKER_OFFSET
    if not MAKER_ONLY:
        return float(p)
    if side.lower() == "buy":
        return float(p * (Decimal("1") - off))
    else:
        return float(p * (Decimal("1") + off))


def build_order_plan(
    signal: str,
    symbol: str,
    equity_usd: float,
    target_fraction: float,
    price: float,
    *,
    lot_step: float | None = None,
    min_qty: float | None = None,
    exch_min_notional: float | None = None
) -> Optional[dict]:
    """
    Build complete order plan using smart sizing logic.
    Returns order dict or None if no order should be placed.
    """
    if signal not in ("BUY", "SELL"):
        return None

    qty, notional, sizing_reason = plan_size(
        equity_usd,
        target_fraction,
        price,
        lot_step=lot_step,
        min_qty=min_qty,
        exch_min_notional=exch_min_notional,
    )

    if qty <= 0:
        return None

    side = "buy" if signal == "BUY" else "sell"
    limit_px = maker_limit_price(side, price)

    order = {
        "symbol": symbol,
        "side": side,
        "qty": float(qty),
        "notional": float(notional),
        "limit_price": limit_px if MAKER_ONLY else None,
        "time_in_force": "post_only" if MAKER_ONLY else "gtc",
        "reason": sizing_reason,
        "maker_only": MAKER_ONLY,
        "maker_offset": float(MAKER_OFFSET) if MAKER_ONLY else 0.0,
    }

    return order
