from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

import ccxt  # type: ignore

from quantbot.config import Settings
from quantbot.state import StateStore


logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class OrderResponse:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    filled_qty: float
    timestamp: str
    info: Dict[str, object]


class BrokerError(Exception):
    """Raised when order placement fails."""


class BaseBroker:
    def get_account(self) -> Dict[str, object]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_position(self, symbol: str) -> Dict[str, object]:  # pragma: no cover
        raise NotImplementedError

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        order_type: str = "market",
        time_in_force: str = "GTC",
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> OrderResponse:  # pragma: no cover
        raise NotImplementedError

    def cancel_open(self, symbol: str) -> None:  # pragma: no cover
        raise NotImplementedError


class PaperBroker(BaseBroker):
    """Simple paper broker that fills orders immediately at the provided price."""

    def __init__(self, state: StateStore, cfg: Settings) -> None:
        self.state = state
        self.settings = cfg
        self.fee_rate = float(cfg.commission)

    def get_account(self) -> Dict[str, object]:
        return self.state.get_account()

    def get_position(self, symbol: str) -> Dict[str, object]:
        return self.state.get_position(symbol)

    def _generate_order_id(self) -> str:
        return uuid4().hex

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        order_type: str = "market",
        time_in_force: str = "GTC",
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> OrderResponse:
        if qty <= 0:
            raise BrokerError("quantity must be positive")

        client_order_id = client_order_id or self._generate_order_id()
        ts = timestamp or _utcnow()

        if existing := self.state.get_order(client_order_id):
            logger.info("duplicate order detected %s for %s", client_order_id, symbol)
            info = dict(existing)
            return OrderResponse(
                order_id=client_order_id,
                symbol=info.get("symbol", symbol),
                side=info.get("side", side),
                qty=float(info.get("qty", qty)),
                price=float(info.get("price", price)),
                status="duplicate",
                filled_qty=float(info.get("filled_qty", info.get("qty", 0.0))),
                timestamp=str(info.get("timestamp", ts)),
                info=info,
            )

        if self.settings.dry_run:
            payload = {
                "order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(price),
                "status": "dry_run",
                "timestamp": ts,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "post_only": bool(post_only),
                "reduce_only": bool(reduce_only),
                "filled_qty": 0.0,
            }
            if side.lower() == "sell":
                payload.setdefault("last_exit_ts", ts)
            self.state.record_order(client_order_id, payload)
            self.state.set_last_action(symbol, side.upper(), ts=ts, order_id=client_order_id, meta=payload)
            return OrderResponse(
                order_id=client_order_id,
                symbol=symbol,
                side=side,
                qty=float(qty),
                price=float(price),
                status="dry_run",
                filled_qty=0.0,
                timestamp=ts,
                info=payload,
            )

        position = self.state.get_position(symbol)
        cash_snapshot = self.state.get_account()
        cash = float(cash_snapshot.get("cash", self.settings.start_cash))
        current_qty = float(position.get("qty", 0.0))
        filled_qty = float(qty)

        if side.lower() == "sell" and current_qty <= 0:
            payload = {
                "order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "qty": 0.0,
                "price": float(price),
                "status": "rejected",
                "timestamp": ts,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "post_only": bool(post_only),
                "reduce_only": bool(reduce_only),
                "reason": "no_position",
                "filled_qty": 0.0,
            }
            self.state.record_order(client_order_id, payload)
            self.state.set_last_action(symbol, "REJECT", ts=ts, order_id=client_order_id, meta=payload)
            return OrderResponse(
                order_id=client_order_id,
                symbol=symbol,
                side=side,
                qty=0.0,
                price=float(price),
                status="rejected",
                filled_qty=0.0,
                timestamp=ts,
                info=payload,
            )

        if side.lower() == "sell":
            filled_qty = min(filled_qty, current_qty)
        notion = price * filled_qty
        fee = notion * self.fee_rate

        if side.lower() == "buy":
            new_qty = current_qty + filled_qty
            prev_val = current_qty * float(position.get("avg_price", 0.0))
            new_avg = (prev_val + notion) / new_qty if new_qty > 0 else 0.0
            cash -= notion + fee
        else:
            new_qty = current_qty - filled_qty
            if new_qty < 0:
                new_qty = 0.0
            new_avg = float(position.get("avg_price", 0.0)) if new_qty > 0 else 0.0
            cash += notion - fee

        self.state.update_position(symbol, new_qty, new_avg)
        self.state.update_account(cash=cash)
        payload = {
            "order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "filled_qty": float(filled_qty),
            "price": float(price),
            "status": "filled",
            "timestamp": ts,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "post_only": bool(post_only),
            "reduce_only": bool(reduce_only),
            "fee": float(fee),
        }
        if side.lower() == "sell":
            payload.setdefault("last_exit_ts", ts)
        self.state.record_order(client_order_id, payload)
        self.state.set_last_action(symbol, side.upper(), ts=ts, order_id=client_order_id, meta=payload)
        return OrderResponse(
            order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=float(qty),
            price=float(price),
            status="filled",
            filled_qty=float(filled_qty),
            timestamp=ts,
            info=payload,
        )

    def cancel_open(self, symbol: str) -> None:
        # Paper broker has no resting orders (fills are immediate)
        logger.debug("cancel_open called for %s (noop in paper mode)", symbol)


class LiveBroker(BaseBroker):
    """Thin ccxt-backed broker for live trading."""

    def __init__(self, cfg: Settings) -> None:
        self.settings = cfg
        exchange_name = (cfg.exchange or cfg.data_provider or "binance").lower()
        if exchange_name == "binance":
            self.exchange = ccxt.binance({
                "apiKey": cfg.live_api_key or cfg.binance_key,
                "secret": cfg.live_api_secret or cfg.binance_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            if cfg.live_base_url:
                self.exchange.urls["api"] = {"public": cfg.live_base_url, "private": cfg.live_base_url}
        elif exchange_name in {"coinbase", "coinbaseadvanced"}:
            self.exchange = ccxt.coinbaseadvanced({
                "apiKey": cfg.live_api_key or cfg.coinbase_key,
                "secret": cfg.live_api_secret or cfg.coinbase_secret,
                "password": cfg.live_api_passphrase or cfg.coinbase_passphrase,
                "enableRateLimit": True,
            })
            if cfg.live_base_url:
                self.exchange.urls["api"] = cfg.live_base_url
        else:
            raise BrokerError(f"Unsupported exchange for live trading: {exchange_name}")
        self.exchange.load_markets()

    def get_account(self) -> Dict[str, object]:
        return self.exchange.fetch_balance()

    def get_position(self, symbol: str) -> Dict[str, object]:
        market_symbol = self.settings.as_ccxt_symbol(symbol)
        balance = self.exchange.fetch_balance()
        base = market_symbol.split("/")[0]
        qty = 0.0
        if isinstance(balance, dict):
            total = balance.get("total") or {}
            qty = float(total.get(base, 0.0))
        return {"qty": qty, "avg_price": 0.0}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        order_type: str = "market",
        time_in_force: str = "GTC",
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> OrderResponse:
        market_symbol = self.settings.as_ccxt_symbol(symbol)
        params: Dict[str, object] = {}
        if client_order_id:
            params["clientOrderId"] = client_order_id
        if time_in_force:
            params["timeInForce"] = time_in_force
        if post_only:
            params["postOnly"] = True
        if reduce_only:
            params["reduceOnly"] = True
        ccxt_price = price if order_type.lower() != "market" else None
        order = self.exchange.create_order(market_symbol, order_type, side, qty, ccxt_price, params)
        ts = timestamp or str(order.get("datetime") or _utcnow())
        return OrderResponse(
            order_id=str(order.get("id") or client_order_id or uuid4().hex),
            symbol=market_symbol,
            side=side,
            qty=float(order.get("amount", qty)),
            price=float(order.get("price", price or 0.0)),
            status=str(order.get("status", "submitted")),
            filled_qty=float(order.get("filled", 0.0)),
            timestamp=ts,
            info=order,
        )

    def cancel_open(self, symbol: str) -> None:
        market_symbol = self.settings.as_ccxt_symbol(symbol)
        try:
            open_orders = self.exchange.fetch_open_orders(symbol=market_symbol)
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Failed to fetch open orders: %s", exc)
            return
        for order in open_orders:
            try:
                order_id = order.get("id")
                if order_id:
                    self.exchange.cancel_order(order_id, symbol=market_symbol)
            except Exception as exc:  # pragma: no cover - network errors
                logger.error("Failed to cancel order %s: %s", order.get("id"), exc)


class BrokerFactory:
    @staticmethod
    def create(state: StateStore, cfg: Settings) -> BaseBroker:
        if cfg.is_live and not cfg.dry_run:
            return LiveBroker(cfg)
        return PaperBroker(state, cfg)


__all__ = [
    "BaseBroker",
    "PaperBroker",
    "LiveBroker",
    "BrokerFactory",
    "OrderResponse",
    "BrokerError",
]
