import os
import json
import random
from datetime import datetime, timezone

import boto3


BUCKET = os.environ.get("S3_BUCKET_TRADES", "")
SYMBOLS = os.environ.get("SYMBOLS", "BTC-USD,ETH-USD").split(",")
TF = os.environ.get("TF", "1h")
REGION = os.environ.get("AWS_REGION", "us-east-2")


def run_bot_once() -> int:
    """Stub: write a run artifact to S3 to validate infra.
    Replace contents with real MACD+ADX logic when ready.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []
    for sym in SYMBOLS:
        pnl = round(random.uniform(-0.005, 0.010), 6)
        results.append(
            {"symbol": sym, "tf": TF, "ts": now, "pnl": pnl, "note": "hello from cloud"}
        )

    key = f"runs/{now.replace(':','').replace('-','')}_{TF}.json"
    body = json.dumps({"ts": now, "symbols": SYMBOLS, "tf": TF, "results": results}, indent=2)
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"))
    print(json.dumps({"level": "info", "msg": "wrote run artifact", "bucket": BUCKET, "key": key}))
    return 0


if __name__ == "__main__":
    assert BUCKET, "S3_BUCKET_TRADES env var is required"
    raise SystemExit(run_bot_once())
