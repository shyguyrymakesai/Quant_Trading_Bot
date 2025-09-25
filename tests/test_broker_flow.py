from pathlib import Path
import sys
import types

if "ccxt" not in sys.modules:
    sys.modules["ccxt"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from types import SimpleNamespace

from quantbot.broker import PaperBroker
from quantbot.state import StateStore


def test_paper_broker_post_only_entry_and_market_exit(tmp_path):
    state_path = tmp_path / "paper_state.json"
    store = StateStore(state_path, initial_cash=100_000.0)
    cfg = SimpleNamespace(commission=0.001, dry_run=True)
    broker = PaperBroker(store, cfg)

    buy_resp = broker.place_order(
        "ETH/USDT",
        "buy",
        qty=1.0,
        price=2000.0,
        order_type="limit",
        time_in_force="GTC",
        post_only=True,
        reduce_only=False,
        client_order_id="entry1",
        timestamp="2024-01-01T00:00:00+00:00",
    )
    assert buy_resp.info["order_type"] == "limit"
    assert buy_resp.info["post_only"] is True

    broker.place_order(
        "ETH/USDT",
        "sell",
        qty=1.0,
        price=2010.0,
        order_type="market",
        time_in_force="GTC",
        post_only=False,
        reduce_only=True,
        client_order_id="exit1",
        timestamp="2024-01-01T01:00:00+00:00",
    )
    last_action = store.get_last_action("ETH/USDT")
    assert last_action["action"] == "SELL"
    assert last_action["meta"]["order_type"] == "market"
    assert last_action["meta"].get("last_exit_ts") == "2024-01-01T01:00:00+00:00"
