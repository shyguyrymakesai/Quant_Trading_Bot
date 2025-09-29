import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from quantbot.scheduler import start_scheduler, tick

async def main():
    # Kick one immediate run, then start scheduler
    await tick()
    start_scheduler()
    # Keep process alive
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down bot...")

if __name__ == "__main__":
    asyncio.run(main())
