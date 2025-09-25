from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class StateStore:
    """Persistent journal for idempotent daemon execution."""

    def __init__(self, path: str | Path, initial_cash: float = 0.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_cash = float(initial_cash)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._persist(self._default_state())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_state(self) -> Dict[str, Any]:
        now = _utcnow()
        return {
            "account": {
                "cash": self.initial_cash,
                "equity": self.initial_cash,
                "start_cash": self.initial_cash,
                "updated": now,
            },
            "positions": {},
            "orders": {},
            "order_history": [],
            "last_actions": {},
            "market_prices": {},
            "run_lock": {},
            "last_processed": {},
            "cycles": [],
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            # Corrupted state → reset but keep backup
            backup = self.path.with_suffix(".corrupt")
            self.path.replace(backup)
            return self._default_state()

    def _persist(self, data: Dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    # ------------------------------------------------------------------
    # Account + position helpers
    # ------------------------------------------------------------------
    def get_account(self) -> Dict[str, Any]:
        with self._lock:
            data = self._load()
            account = data.setdefault("account", {
                "cash": self.initial_cash,
                "equity": self.initial_cash,
                "start_cash": self.initial_cash,
                "updated": _utcnow(),
            })
            positions = data.setdefault("positions", {})
            prices = data.setdefault("market_prices", {})
            equity = float(account.get("cash", self.initial_cash))
            for sym, pos in positions.items():
                qty = float(pos.get("qty", 0.0))
                mark = float(prices.get(sym, pos.get("avg_price", 0.0)))
                equity += qty * mark
            account["equity"] = equity
            account["updated"] = _utcnow()
            self._persist(data)
            return json.loads(json.dumps(account))  # deep copy

    def update_account(self, *, cash: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            data = self._load()
            account = data.setdefault("account", {
                "cash": self.initial_cash,
                "equity": self.initial_cash,
                "start_cash": self.initial_cash,
                "updated": _utcnow(),
            })
            if cash is not None:
                account["cash"] = float(cash)
            self._persist(data)
            return json.loads(json.dumps(account))

    def get_position(self, symbol: str) -> Dict[str, Any]:
        with self._lock:
            data = self._load()
            pos = data.setdefault("positions", {}).get(symbol)
            if pos is None:
                return {"qty": 0.0, "avg_price": 0.0, "updated": None}
            return json.loads(json.dumps(pos))

    def update_position(self, symbol: str, qty: float, avg_price: float) -> None:
        with self._lock:
            data = self._load()
            positions = data.setdefault("positions", {})
            positions[symbol] = {
                "qty": float(qty),
                "avg_price": float(avg_price),
                "updated": _utcnow(),
            }
            self._persist(data)

    def update_market_price(self, symbol: str, price: float) -> None:
        with self._lock:
            data = self._load()
            market_prices = data.setdefault("market_prices", {})
            market_prices[symbol] = {
                "price": float(price),
                "updated": _utcnow(),
            }
            self._persist(data)

    # ------------------------------------------------------------------
    # Order + journal helpers
    # ------------------------------------------------------------------
    def order_exists(self, client_order_id: str) -> bool:
        with self._lock:
            data = self._load()
            orders = data.setdefault("orders", {})
            return client_order_id in orders

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load()
            orders = data.setdefault("orders", {})
            order = orders.get(client_order_id)
            if order is None:
                return None
            return json.loads(json.dumps(order))

    def record_order(self, client_order_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            orders = data.setdefault("orders", {})
            history = data.setdefault("order_history", [])
            orders[client_order_id] = payload
            history.append(payload)
            # Keep journal size reasonable
            if len(history) > 5000:
                history[:] = history[-5000:]
            self._persist(data)

    def get_last_action(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load()
            actions = data.setdefault("last_actions", {})
            return actions.get(symbol)

    def set_last_action(
        self,
        symbol: str,
        action: str,
        *,
        ts: str,
        order_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            data = self._load()
            actions = data.setdefault("last_actions", {})
            actions[symbol] = {
                "action": action,
                "ts": ts,
                "order_id": order_id,
                "meta": meta or {},
            }
            self._persist(data)

    def record_cycle(self, symbol: str, candle: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            cycles = data.setdefault("cycles", [])
            cycles.append({
                "symbol": symbol,
                "candle": candle,
                "payload": payload,
                "recorded": _utcnow(),
            })
            if len(cycles) > 2000:
                cycles[:] = cycles[-2000:]
            self._persist(data)

    # ------------------------------------------------------------------
    # Run lock for idempotent execution
    # ------------------------------------------------------------------
    @contextmanager
    def candle_lock(self, symbol: str, candle: str) -> Iterator[bool]:
        """Context manager returning True if lock acquired for this candle."""

        with self._lock:
            data = self._load()
            run_lock = data.setdefault("run_lock", {})
            last_processed = data.setdefault("last_processed", {})
            info = run_lock.get(symbol)
            if info:
                if info.get("candle") == candle and info.get("status") in {"running", "done"}:
                    yield False
                    return
            if last_processed.get(symbol) == candle:
                yield False
                return
            run_lock[symbol] = {
                "candle": candle,
                "status": "running",
                "ts": _utcnow(),
            }
            self._persist(data)

        try:
            yield True
        except Exception:
            with self._lock:
                data = self._load()
                run_lock = data.setdefault("run_lock", {})
                run_lock[symbol] = {
                    "candle": candle,
                    "status": "error",
                    "ts": _utcnow(),
                }
                self._persist(data)
            raise
        else:
            with self._lock:
                data = self._load()
                run_lock = data.setdefault("run_lock", {})
                run_lock[symbol] = {
                    "candle": candle,
                    "status": "done",
                    "ts": _utcnow(),
                }
                data.setdefault("last_processed", {})[symbol] = candle
                self._persist(data)

    # ------------------------------------------------------------------
    # Utility helpers (mainly for tests)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._persist(self._default_state())


__all__ = ["StateStore"]
