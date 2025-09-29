import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from quantbot.scheduler import start_scheduler, tick


async def main():
    # Run once and exit (EventBridge handles scheduling)
    await tick()
    print("Bot run completed.")


if __name__ == "__main__":
    asyncio.run(main())
