import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from quantbot.scheduler import start_scheduler, tick

if __name__ == "__main__":
    # Kick one immediate run, then start scheduler
    asyncio.run(tick())
    start_scheduler()
    # Keep process alive
    import time

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down bot...")
