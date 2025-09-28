#!/usr/bin/env python3
"""Simple debug script to check environment in container."""

import os
import json
from pathlib import Path


def main():
    print("=== DEBUG: Environment Variables ===")
    relevant_vars = [
        "ENV",
        "MODE",
        "env",
        "mode",
        "PYTHONPATH",
        "AWS_REGION",
        "COINBASE_API_KEY",
    ]
    for var in relevant_vars:
        value = os.environ.get(var, "NOT SET")
        if var.startswith("COINBASE"):
            print(f"{var}: {'SET' if value != 'NOT SET' else 'NOT SET'}")
        else:
            print(f"{var}: {value}")

    print("\n=== DEBUG: Files ===")
    env_file = Path(".env")
    if env_file.exists():
        print(f".env exists: {env_file.resolve()}")
        with open(env_file) as f:
            content = f.read()
            print("--- .env content ---")
            print(content[:200])
            print("--- end ---")
    else:
        print(".env does not exist")

    try:
        from src.config import Settings

        cfg = Settings()
        print(f"\n=== DEBUG: Pydantic ===")
        print(f"cfg.env: {cfg.env}")
        print(f"cfg.mode: {cfg.mode}")
        print(f"cfg.is_live: {cfg.is_live}")
    except Exception as e:
        print(f"\nError loading config: {e}")


if __name__ == "__main__":
    main()
