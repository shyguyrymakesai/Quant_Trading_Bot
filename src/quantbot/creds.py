"""
Coinbase credentials loader from AWS SSM Parameter Store.
Securely loads API keys at runtime without storing in plaintext.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PFX = os.getenv("COINBASE_PARAM_PREFIX", "/quant-bot/coinbase")
REGION = os.getenv("AWS_REGION", "us-east-1")


def load_coinbase_secrets() -> Optional[Dict[str, str]]:
    """
    Load Coinbase API credentials from AWS SSM Parameter Store.

    Returns:
        Dict with api_key, api_secret, and optionally passphrase
        None if unable to load (logs error)
    """
    try:
        import boto3

        ssm = boto3.client("ssm", region_name=REGION)

        mode = os.getenv("COINBASE_API_MODE", "advanced_trade")

        if mode == "advanced_trade":
            # Coinbase Advanced Trade API (retail)
            names = [f"{PFX}/API_KEY", f"{PFX}/API_SECRET"]
            logger.info(f"Loading Advanced Trade credentials from SSM: {names}")

            resp = ssm.get_parameters(Names=names, WithDecryption=True)
            if len(resp["Parameters"]) != len(names):
                missing = set(names) - {p["Name"] for p in resp["Parameters"]}
                logger.error(f"Missing SSM parameters: {missing}")
                return None

            vals = {p["Name"].split("/")[-1]: p["Value"] for p in resp["Parameters"]}
            return {
                "mode": mode,
                "api_key": vals["API_KEY"],
                "api_secret": vals["API_SECRET"],
            }

        elif mode == "exchange":
            # Coinbase Exchange API (institutional, has passphrase)
            names = [f"{PFX}/EX_KEY", f"{PFX}/EX_SECRET", f"{PFX}/EX_PASSPHRASE"]
            logger.info(f"Loading Exchange API credentials from SSM: {names}")

            resp = ssm.get_parameters(Names=names, WithDecryption=True)
            if len(resp["Parameters"]) != len(names):
                missing = set(names) - {p["Name"] for p in resp["Parameters"]}
                logger.error(f"Missing SSM parameters: {missing}")
                return None

            vals = {p["Name"].split("/")[-1]: p["Value"] for p in resp["Parameters"]}
            return {
                "mode": mode,
                "api_key": vals["EX_KEY"],
                "api_secret": vals["EX_SECRET"],
                "passphrase": vals["EX_PASSPHRASE"],
            }
        else:
            logger.error(f"Unknown COINBASE_API_MODE: {mode}")
            return None

    except ImportError:
        logger.error("boto3 not available - cannot load SSM credentials")
        return None
    except Exception as exc:
        logger.error(f"Failed to load Coinbase credentials from SSM: {exc}")
        return None


def get_live_credentials() -> Dict[str, Optional[str]]:
    """
    Get live trading credentials, falling back to environment variables.

    Returns:
        Dict with live_api_key, live_api_secret, live_api_passphrase
    """
    # First try SSM
    creds = load_coinbase_secrets()
    if creds:
        logger.info(f"Loaded {creds['mode']} credentials from SSM")
        return {
            "live_api_key": creds["api_key"],
            "live_api_secret": creds["api_secret"],
            "live_api_passphrase": creds.get("passphrase"),
        }

    # Fallback to environment variables
    logger.warning("SSM credentials not available, using environment variables")
    return {
        "live_api_key": os.getenv("COINBASE_API_KEY"),
        "live_api_secret": os.getenv("COINBASE_SECRET"),
        "live_api_passphrase": os.getenv("COINBASE_PASSPHRASE"),
    }
