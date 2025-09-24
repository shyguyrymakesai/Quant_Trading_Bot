import os
import csv
from typing import Dict, List

# Reuse project root resolver from config for consistent paths
from .config import _project_root


def _logs_dir() -> str:
    d = os.path.join(_project_root(), "logs")
    os.makedirs(d, exist_ok=True)
    return d


def append_row(filename: str, fieldnames: List[str], row: Dict):
    path = os.path.join(_logs_dir(), filename)
    file_exists = os.path.exists(path)
    write_header = True
    if file_exists:
        try:
            write_header = os.path.getsize(path) == 0
        except OSError:
            write_header = True
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_paper_trade(row: Dict):
    fields = [
        "timestamp",
        "exchange",
        "mode",
        "symbol",
        "side",
        "qty",
        "price",
        "last_close",
        "realized_vol",
        "size_scale",
        "spread_bps",
        "status",
        "rationale",
    ]
    append_row("paper_trades.csv", fields, row)
