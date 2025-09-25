from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_TZ = "America/Indiana/Indianapolis"


class Settings(BaseSettings):
    """Runtime configuration loaded from YAML + environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # environment + orchestration
    env: str = "paper"
    mode: str = "paper"
    dry_run: bool = True
    timezone: str = DEFAULT_TZ
    tz: str | None = None
    exchange: str = "binance"
    data_provider: str = "binance"
    symbols: List[str] = Field(default_factory=list)
    symbol: str = "BTC/USDT"
    timeframe: str = "30m"
    base_currency: str = "USDT"
    quote_currency: str = "USDT"
    lookback_limit: int = 360
    processing_lag_seconds: int = 20
    misfire_grace_time: int = 300
    max_instances: int = 1
    max_api_errors: int = 5
    circuit_breaker_cooldown_minutes: int = 30
    state_file: str = "state/bot_state.json"
    log_dir: str = "logs"
    log_level: str = "INFO"
    start_cash: float = 100_000.0
    max_leverage: float = 1.0
    notional_buffer: float = 0.0
    min_order_notional: float = 0.0
    min_order_qty: float = 0.0

    # strategy params (defaults from MACD sweep)
    macd_fast: int = 8
    macd_slow: int = 24
    macd_signal: int = 9
    adx_len: int = 14
    adx_threshold: float = 20.0
    vol_lookback: int = 20
    vol_target: float = 0.02
    min_size: float = 0.0
    max_size: float = 1.0
    cooldown_bars: int = 0
    entry_order_type: str = "market"
    exit_order_type: str = "market"
    maker_offset_bps: float = 1.0

    # Consolidated per-symbol overrides (single source of truth)
    # Keys may be like "BTC-USD", "BTC/USD", or compact forms; we normalize.
    strategy_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # legacy risk fields (retained for backwards compat)
    fees_taker: float = 0.001
    slippage_bps: int = 2
    daily_loss_cap_pct: float = 0.01
    max_position_notional: float = 25.0
    commission: float = 0.0012
    risk_frac: float = 0.10
    max_pos_frac: float = 1.0
    daily_loss_limit: float = -0.02
    spread_bps_limit: int = 25
    risk_slippage_bps: int = 10

    # API credentials (paper + live)
    binance_key: str | None = None
    binance_secret: str | None = None
    coinbase_key: str | None = None
    coinbase_secret: str | None = None
    coinbase_passphrase: str | None = None

    paper_api_key: str | None = None
    paper_api_secret: str | None = None
    paper_api_passphrase: str | None = None
    paper_base_url: str | None = None
    live_api_key: str | None = None
    live_api_secret: str | None = None
    live_api_passphrase: str | None = None
    live_base_url: str | None = None

    @field_validator("symbols", mode="before")
    @classmethod
    def _parse_symbols(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            if value.strip().startswith("["):
                # JSON / YAML style list string
                try:
                    import json

                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [
                            str(item).strip() for item in parsed if str(item).strip()
                        ]
                except Exception:
                    pass
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()]

    @model_validator(mode="after")
    def _sync_fields(self) -> "Settings":
        # Normalise env/mode naming
        normalized_mode = (self.mode or self.env or "paper").lower()
        self.mode = normalized_mode
        self.env = normalized_mode
        if not self.symbols:
            self.symbols = [self.symbol]
        if not self.symbol:
            self.symbol = self.symbols[0]
        if self.tz is None:
            self.tz = self.timezone
        return self

    @property
    def symbol_universe(self) -> List[str]:
        return list(self.symbols)

    def as_ccxt_symbol(self, symbol: str | None = None) -> str:
        sym = symbol or self.symbol
        if "/" in sym:
            return sym
        sym = sym.replace("-", "/").replace("_", "/")
        if "/" in sym:
            return sym
        quote = (self.quote_currency or "").replace("/", "").upper()
        if quote and sym.upper().endswith(quote):
            base = sym[: -len(quote)]
            return f"{base}/{quote}"
        # Best effort split (common stablecoins 3-4 chars)
        if len(sym) > 6:
            return f"{sym[:-4]}/{sym[-4:]}"
        return sym

    @property
    def is_live(self) -> bool:
        return self.env.lower() == "live"

    @property
    def timezone_name(self) -> str:
        return self.tz or self.timezone or DEFAULT_TZ


def _project_root() -> Path:
    """Best-effort project root discovery.
    We search upward from this file for a folder that contains `config/config.yaml`
    or a `config` directory. Fallback to the package parent.
    """
    here = Path(__file__).resolve()
    for p in [
        here.parent,
        here.parent.parent,
        here.parent.parent.parent,
        here.parent.parent.parent.parent,
    ]:
        try:
            if (p / "config" / "config.yaml").exists() or (p / "config").exists():
                return p
        except Exception:
            continue
    return here.parent.parent


def load_yaml_settings(path: str | None = None) -> Dict[str, Any]:
    env_path = os.getenv("CONFIG_PATH")
    cfg_path = Path(path or env_path or _project_root() / "config" / "config.yaml")
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


yaml_cfg = load_yaml_settings()


def _flatten_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not cfg:
        return out
    out["mode"] = cfg.get("mode", cfg.get("env", "paper"))
    out["env"] = cfg.get("mode", cfg.get("env", "paper"))
    out["exchange"] = cfg.get("exchange", "binance")
    out["symbol"] = cfg.get(
        "symbol",
        cfg.get("symbols", ["BTC/USDT"])[0] if cfg.get("symbols") else "BTC/USDT",
    )
    if "symbols" in cfg:
        out["symbols"] = cfg["symbols"]
    out["timeframe"] = cfg.get("timeframe", "30m")
    out["timezone"] = cfg.get("timezone", DEFAULT_TZ)
    out["tz"] = cfg.get("timezone", DEFAULT_TZ)
    out["data_provider"] = cfg.get("data_provider", cfg.get("exchange", "binance"))
    out["base_currency"] = cfg.get("base_currency", "USDT")
    out["quote_currency"] = cfg.get("quote_currency", cfg.get("base_currency", "USDT"))
    out["lookback_limit"] = cfg.get("lookback_limit", 360)
    out["processing_lag_seconds"] = cfg.get("processing_lag_seconds", 20)
    out["misfire_grace_time"] = cfg.get("misfire_grace_time", 300)
    out["max_instances"] = cfg.get("max_instances", 1)
    out["max_api_errors"] = cfg.get("max_api_errors", 5)
    out["circuit_breaker_cooldown_minutes"] = cfg.get(
        "circuit_breaker_cooldown_minutes", 30
    )
    out["state_file"] = cfg.get(
        "state_file", cfg.get("state", {}).get("path", "state/bot_state.json")
    )
    out["log_dir"] = cfg.get("logging", {}).get("dir", cfg.get("log_dir", "logs"))
    out["log_level"] = cfg.get("logging", {}).get("level", cfg.get("log_level", "INFO"))
    out["start_cash"] = cfg.get("risk", {}).get(
        "start_cash", cfg.get("start_cash", 100_000.0)
    )
    out["max_leverage"] = cfg.get("risk", {}).get(
        "max_leverage", cfg.get("max_leverage", 1.0)
    )
    out["notional_buffer"] = cfg.get("risk", {}).get(
        "notional_buffer", cfg.get("notional_buffer", 0.0)
    )
    out["min_order_notional"] = cfg.get("risk", {}).get(
        "min_order_notional", cfg.get("min_order_notional", 0.0)
    )
    out["min_order_qty"] = cfg.get("risk", {}).get(
        "min_order_qty", cfg.get("min_order_qty", 0.0)
    )
    strat_cfg = cfg.get("strategy", {})
    macd_cfg = strat_cfg.get("macd", {})
    adx_cfg = strat_cfg.get("adx", {})
    vol_cfg = strat_cfg.get("vol_target", {})
    out["macd_fast"] = macd_cfg.get("fast", 8)
    out["macd_slow"] = macd_cfg.get("slow", 24)
    out["macd_signal"] = macd_cfg.get("signal", 9)
    out["adx_len"] = adx_cfg.get("length", 14)
    out["adx_threshold"] = adx_cfg.get("threshold", 20)
    out["vol_lookback"] = vol_cfg.get("lookback", 20)
    out["vol_target"] = vol_cfg.get("target_vol", 0.02)
    out["min_size"] = vol_cfg.get("min_size", 0.0)
    out["max_size"] = vol_cfg.get("max_size", 1.0)
    out["cooldown_bars"] = strat_cfg.get("cooldown_bars", cfg.get("cooldown_bars", 0))
    orders_cfg = strat_cfg.get("orders", {})
    out["entry_order_type"] = orders_cfg.get(
        "entry",
        strat_cfg.get("entry_order_type", cfg.get("entry_order_type", "market")),
    )
    out["exit_order_type"] = orders_cfg.get(
        "exit", strat_cfg.get("exit_order_type", cfg.get("exit_order_type", "market"))
    )
    out["maker_offset_bps"] = orders_cfg.get(
        "maker_offset_bps", cfg.get("maker_offset_bps", 1.0)
    )

    risk_cfg = cfg.get("risk", {})
    out["commission"] = risk_cfg.get("commission", 0.0012)
    out["risk_frac"] = risk_cfg.get("risk_frac", 0.10)
    out["max_pos_frac"] = risk_cfg.get("max_pos_frac", 1.0)
    out["daily_loss_limit"] = risk_cfg.get("daily_loss_limit", -0.02)
    out["spread_bps_limit"] = risk_cfg.get("spread_bps_limit", 25)
    out["risk_slippage_bps"] = risk_cfg.get("slippage_bps", 10)

    api_cfg = cfg.get("api_keys", {})
    out["binance_key"] = api_cfg.get("binance", {}).get("key")
    out["binance_secret"] = api_cfg.get("binance", {}).get("secret")
    out["coinbase_key"] = api_cfg.get("coinbase", {}).get("key")
    out["coinbase_secret"] = api_cfg.get("coinbase", {}).get("secret")
    out["coinbase_passphrase"] = api_cfg.get("coinbase", {}).get("passphrase")
    out["paper_api_key"] = api_cfg.get("paper", {}).get("key")
    out["paper_api_secret"] = api_cfg.get("paper", {}).get("secret")
    out["paper_api_passphrase"] = api_cfg.get("paper", {}).get("passphrase")
    out["paper_base_url"] = api_cfg.get("paper", {}).get("base_url")
    out["live_api_key"] = api_cfg.get("live", {}).get("key")
    out["live_api_secret"] = api_cfg.get("live", {}).get("secret")
    out["live_api_passphrase"] = api_cfg.get("live", {}).get("passphrase")
    out["live_base_url"] = api_cfg.get("live", {}).get("base_url")
    # Consolidated per-symbol overrides live under strategy_overrides
    if "strategy_overrides" in cfg and isinstance(cfg["strategy_overrides"], dict):
        out["strategy_overrides"] = cfg["strategy_overrides"]
    return out


settings = Settings(**_flatten_yaml(yaml_cfg))


def _normalize_symbol_key(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").replace("_", "").upper()


def _iter_symbol_aliases(symbol_ccxt: str) -> List[str]:
    """Generate common aliases for a symbol to match keys in configs.
    e.g., BTC/USDT -> [BTCUSDT, BTC-USD, BTC/USD, BTCUSD]
    """
    s = symbol_ccxt or ""
    if not s:
        return []
    norm = _normalize_symbol_key(s)
    aliases = {norm}
    try:
        base, quote = s.replace("-", "/").replace("_", "/").split("/")
    except ValueError:
        base, quote = s, "USD"
    # CCXT style
    aliases.add(f"{base}/{quote}")
    # Dash style
    aliases.add(f"{base}-{quote}")
    # Compact common
    aliases.add(f"{base}{quote}")
    # USD override if USDT
    if quote.upper() == "USDT":
        aliases.add(f"{base}-USD")
        aliases.add(f"{base}/USD")
        aliases.add(f"{base}USD")
    # Also add uppercase only keys
    aliases.update({a.upper() for a in list(aliases)})
    return list(aliases)


def get_symbol_params(symbol_ccxt: str) -> Dict[str, Any]:
    """Return per-symbol parameter overrides from YAML `strategy_overrides` only."""
    if not symbol_ccxt:
        return {}
    aliases = _iter_symbol_aliases(symbol_ccxt)

    # YAML consolidated overrides
    try:
        overrides = getattr(settings, "strategy_overrides", {}) or {}
        if isinstance(overrides, dict):
            # allow both key forms and nested dicts
            for a in aliases:
                if a in overrides and isinstance(overrides[a], dict):
                    return dict(overrides[a])
            # try normalized keys
            over_norm = {
                _normalize_symbol_key(k): v
                for k, v in overrides.items()
                if isinstance(v, dict)
            }
            for a in aliases:
                na = _normalize_symbol_key(a)
                if na in over_norm:
                    return dict(over_norm[na])
    except Exception:
        pass
    return {}
