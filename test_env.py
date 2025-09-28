#!/usr/bin/env python3
"""Test environment variable loading."""

import os
from src.config import Settings


def main():
    print("=== Environment Variables ===")
    env_vars = ["ENV", "MODE", "env", "mode"]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'NOT SET')}")

    print("\n=== Pydantic Settings ===")
    cfg = Settings()
    print(f"cfg.env: {cfg.env}")
    print(f"cfg.mode: {cfg.mode}")
    print(f"cfg.is_live: {cfg.is_live}")

    print(f"\nNormalized mode logic:")
    print(f"(cfg.mode or cfg.env or 'paper'): {cfg.mode or cfg.env or 'paper'}")


if __name__ == "__main__":
    main()
