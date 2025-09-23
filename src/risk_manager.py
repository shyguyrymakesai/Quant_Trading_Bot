from .config import settings

def cap_position_size(notional: float) -> float:
    return min(notional, settings.max_position_notional)

def daily_loss_guard(current_day_pnl_pct: float) -> bool:
    """Return True if trading should HALT for the day."""
    return current_day_pnl_pct <= -settings.daily_loss_cap_pct
