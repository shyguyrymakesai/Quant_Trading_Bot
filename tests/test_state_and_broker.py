from pathlib import Path
import sys
import types

if 'ccxt' not in sys.modules:
    sys.modules['ccxt'] = types.SimpleNamespace()

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from src.broker import PaperBroker
from src.config import Settings
from src.state import StateStore


def test_candle_lock_idempotent(tmp_path):
    state_path = tmp_path / "state.json"
    store = StateStore(state_path, initial_cash=1000.0)

    with store.candle_lock("BTC/USDT", "2024-01-01T00:30:00Z") as first:
        assert first is True
    with store.candle_lock("BTC/USDT", "2024-01-01T00:30:00Z") as second:
        assert second is False


def test_paper_broker_execution_and_duplicate(tmp_path):
    state_path = tmp_path / "broker_state.json"
    cfg = Settings(
        env="paper",
        dry_run=False,
        start_cash=10000.0,
        commission=0.0,
        symbols=["BTC/USDT"],
        symbol="BTC/USDT",
    )
    store = StateStore(state_path, initial_cash=cfg.start_cash)
    broker = PaperBroker(store, cfg)

    order = broker.place_order(
        "BTC/USDT",
        "buy",
        qty=1.0,
        price=100.0,
        client_order_id="test-123",
        timestamp="2024-01-01T00:30:00Z",
    )
    assert order.status == "filled"
    account = broker.get_account()
    assert account["cash"] < cfg.start_cash
    position = broker.get_position("BTC/USDT")
    assert pytest.approx(position["qty"], rel=1e-6) == 1.0

    duplicate = broker.place_order(
        "BTC/USDT",
        "buy",
        qty=1.0,
        price=100.0,
        client_order_id="test-123",
        timestamp="2024-01-01T00:30:00Z",
    )
    assert duplicate.status == "duplicate"
    # No additional quantity added
    position_after = broker.get_position("BTC/USDT")
    assert pytest.approx(position_after["qty"], rel=1e-6) == 1.0
