from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # boto3 is only required when uploading to S3
    import boto3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None

from quantbot.bot_daemon import BotDaemon, _setup_logging
from quantbot.config import Settings, load_config_file, settings


logger = logging.getLogger("quantbot.run_once")


def _load_state_snapshot(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        logger.warning(
            "State file %s is not valid JSON; using empty snapshot", state_path
        )
        return {}


def _latest_cycle(
    cycles: List[Dict[str, Any]], symbol: str
) -> Optional[Dict[str, Any]]:
    for entry in reversed(cycles):
        if entry.get("symbol") == symbol:
            return entry
    return None


def _build_artifact(cfg: Settings, state_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg.timeframe}"
    cycles = list(state_snapshot.get("cycles", []) or [])
    last_actions = state_snapshot.get("last_actions", {}) or {}
    positions = state_snapshot.get("positions", {}) or {}
    account = state_snapshot.get("account", {}) or {}
    symbols_summary: List[Dict[str, Any]] = []
    for symbol in cfg.symbol_universe:
        summary: Dict[str, Any] = {"symbol": symbol}
        latest = _latest_cycle(cycles, symbol)
        if latest:
            summary["candle"] = latest.get("candle")
            summary["recorded"] = latest.get("recorded")
            summary["cycle_payload"] = latest.get("payload")
        if symbol in last_actions:
            summary["last_action"] = last_actions.get(symbol)
        if symbol in positions:
            summary["position"] = positions.get(symbol)
        symbols_summary.append(summary)

    orders = state_snapshot.get("order_history", []) or []
    recent_orders = orders[-10:]

    artifact = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "mode": cfg.env,
        "timeframe": cfg.timeframe,
        "exchange": cfg.exchange,
        "data_provider": cfg.data_provider,
        "symbols": symbols_summary,
        "account": account,
        "recent_orders": recent_orders,
        "state_file": str(Path(cfg.state_file).resolve()),
    }
    return artifact


def _write_artifact(artifact: Dict[str, Any], cfg: Settings) -> None:
    body = json.dumps(artifact, indent=2).encode("utf-8")
    bucket = os.environ.get("S3_BUCKET_TRADES")
    object_key = os.environ.get("S3_OBJECT_KEY") or f"runs/{artifact['run_id']}.json"
    region = os.environ.get("AWS_REGION")

    if bucket:
        if boto3 is None:
            raise RuntimeError(
                "boto3 is required to upload artifacts to S3; install deploy requirements or use --skip-upload"
            )
        client_kwargs = {"region_name": region} if region else {}
        s3 = boto3.client("s3", **client_kwargs)
        s3.put_object(Bucket=bucket, Key=object_key, Body=body)
        logger.info("Uploaded run artifact to s3://%s/%s", bucket, object_key)
    else:
        out_dir = Path(cfg.state_file).parent / "run_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / artifact["run_id"].replace(":", "")
        if not out_path.suffix:
            out_path = out_path.with_suffix(".json")
        out_path.write_bytes(body)
        logger.info("S3 bucket not provided; wrote artifact to %s", out_path)


async def _execute_cycle(bot: BotDaemon) -> None:
    # For run-once execution we do not need to sleep before processing.
    bot.cfg.processing_lag_seconds = 0
    await bot.run_cycle()
    await bot.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single trading cycle and exit")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Optional path to a run-config YAML file",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading the artifact to S3 (writes locally only)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg: Settings
    if args.config_path:
        cfg = load_config_file(args.config_path)
    else:
        cfg = settings

    _setup_logging(cfg)
    logger.info(
        "Starting single-cycle run for symbols %s on %s (%s mode)",
        ", ".join(cfg.symbol_universe),
        cfg.exchange,
        cfg.env,
    )

    bot = BotDaemon(cfg)
    asyncio.run(_execute_cycle(bot))

    state_snapshot = _load_state_snapshot(Path(cfg.state_file))
    artifact = _build_artifact(cfg, state_snapshot)
    if args.skip_upload:
        os.environ.pop("S3_BUCKET_TRADES", None)
    _write_artifact(artifact, cfg)
    logger.info("Run complete: %s", artifact["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
