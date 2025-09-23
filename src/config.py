import os, yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mode: str = "paper"
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    base_currency: str = "USDT"

    # fees
    fees_taker: float = 0.001
    slippage_bps: int = 2

    # risk
    daily_loss_cap_pct: float = 0.01
    max_position_notional: float = 25.0

    # strategy
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_len: int = 14
    adx_threshold: int = 25
    vol_lookback: int = 20
    vol_target: float = 0.02
    min_size: float = 0.0
    max_size: float = 1.0

    # api keys
    binance_key: str | None = None
    binance_secret: str | None = None
    coinbase_key: str | None = None
    coinbase_secret: str | None = None
    coinbase_passphrase: str | None = None

    class Config:
        env_file = ".env"

def load_yaml_settings(path: str = "config/config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

yaml_cfg = load_yaml_settings()
settings = Settings(**{ 
    # map yaml keys to Settings fields where names differ
    "mode": yaml_cfg.get("mode", "paper"),
    "exchange": yaml_cfg.get("exchange", "binance"),
    "symbol": yaml_cfg.get("symbol", "BTC/USDT"),
    "timeframe": yaml_cfg.get("timeframe", "1h"),
    "base_currency": yaml_cfg.get("base_currency", "USDT"),
    "fees_taker": yaml_cfg.get("fees", {}).get("taker", 0.001),
    "slippage_bps": yaml_cfg.get("fees", {}).get("slippage_bps", 2),
    "daily_loss_cap_pct": yaml_cfg.get("risk", {}).get("daily_loss_cap_pct", 0.01),
    "max_position_notional": yaml_cfg.get("risk", {}).get("max_position_notional", 25.0),
    "macd_fast": yaml_cfg.get("strategy", {}).get("macd", {}).get("fast", 12),
    "macd_slow": yaml_cfg.get("strategy", {}).get("macd", {}).get("slow", 26),
    "macd_signal": yaml_cfg.get("strategy", {}).get("macd", {}).get("signal", 9),
    "adx_len": yaml_cfg.get("strategy", {}).get("adx", {}).get("length", 14),
    "adx_threshold": yaml_cfg.get("strategy", {}).get("adx", {}).get("threshold", 25),
    "vol_lookback": yaml_cfg.get("strategy", {}).get("vol_target", {}).get("lookback", 20),
    "vol_target": yaml_cfg.get("strategy", {}).get("vol_target", {}).get("target_vol", 0.02),
    "min_size": yaml_cfg.get("strategy", {}).get("vol_target", {}).get("min_size", 0.0),
    "max_size": yaml_cfg.get("strategy", {}).get("vol_target", {}).get("max_size", 1.0),
    "binance_key": yaml_cfg.get("api_keys", {}).get("binance", {}).get("key"),
    "binance_secret": yaml_cfg.get("api_keys", {}).get("binance", {}).get("secret"),
    "coinbase_key": yaml_cfg.get("api_keys", {}).get("coinbase", {}).get("key"),
    "coinbase_secret": yaml_cfg.get("api_keys", {}).get("coinbase", {}).get("secret"),
    "coinbase_passphrase": yaml_cfg.get("api_keys", {}).get("coinbase", {}).get("passphrase"),
})
