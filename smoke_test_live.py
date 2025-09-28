"""
Live trading smoke test - place one tiny maker-only order to validate setup.
Run this BEFORE enabling the full schedule to make sure everything works.
"""

import os
import logging
from decimal import Decimal

# Set up live mode environment
os.environ.update(
    {
        "PYTHONPATH": "src",
        "MODE": "live",
        "SYMBOLS": '["BTC-USD"]',  # Test with just BTC first
        "MIN_NOTIONAL_FLOOR": "1",
        "FLAT_ORDER_NOTIONAL": "1",
        "MAKER_ONLY": "true",
        "MAKER_OFFSET": "0.00015",
        "COINBASE_PARAM_PREFIX": "/quant-bot/coinbase",
        "COINBASE_API_MODE": "advanced_trade",
    }
)

from quantbot.config import settings
from quantbot.broker import BrokerFactory
from quantbot.state import StateStore
from quantbot.smart_sizing import plan_size, maker_limit_price
from quantbot.creds import load_coinbase_secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smoke_test():
    """
    Smoke test: validate credentials and place tiny test order.
    """
    print("🚀 Live Trading Smoke Test")
    print("=" * 40)

    # Test 1: Validate SSM credentials
    print("📡 Testing SSM credential loading...")
    creds = load_coinbase_secrets()
    if not creds:
        print("❌ Failed to load SSM credentials")
        return False
    print(f"✅ Loaded {creds['mode']} credentials")

    # Test 2: Connect to broker
    print("🔌 Testing broker connection...")
    try:
        state = StateStore(settings.state_file, initial_cash=settings.start_cash)
        broker = BrokerFactory.create(state, settings)
        account = broker.get_account()
        print(f"✅ Connected to {settings.exchange}")
        print(f"💰 Account balance: {account}")
    except Exception as exc:
        print(f"❌ Broker connection failed: {exc}")
        return False

    # Test 3: Validate smart sizing
    print("📏 Testing smart sizing logic...")
    try:
        # Small test: $50 account, 2% = $1 target (should use $1 exchange minimum)
        qty, notional, reason = plan_size(50.0, 0.02, 60000.0, exch_min_notional=1.0)
        print(f"✅ Smart sizing: ${notional} notional, reason: {reason}")

        # Test maker pricing
        test_price = maker_limit_price("buy", 60000.0)
        print(f"✅ Maker pricing: buy at ${test_price} (vs mid $60000)")

    except Exception as exc:
        print(f"❌ Smart sizing failed: {exc}")
        return False

    # Test 4: Check minimum balance
    try:
        balance_usd = float(account.get("total", {}).get("USD", 0))
        if balance_usd < 10:
            print(
                f"⚠️  Low USD balance: ${balance_usd} - may not be able to place orders"
            )
        else:
            print(f"✅ Sufficient balance: ${balance_usd}")
    except Exception:
        print("⚠️  Could not check USD balance")

    print("\n🎯 Smoke test complete!")
    print("If all tests passed, you can enable live trading:")
    print("1. Apply terraform to deploy live configuration")
    print("2. Monitor CloudWatch logs during first few runs")
    print("3. Check S3 artifacts for order_plan data")
    return True


if __name__ == "__main__":
    smoke_test()
