import aiosqlite
import os
from pathlib import Path

DB_PATH = os.environ.get("BOT_DB_PATH", "bot_state.sqlite3")

DDL = '''
CREATE TABLE IF NOT EXISTS trades(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL,
    rationale TEXT
);
CREATE TABLE IF NOT EXISTS metrics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    pnl REAL,
    equity REAL,
    drawdown REAL
);
'''

async def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(DDL)
        await db.commit()

async def log_trade(ts, symbol, side, qty, price, fee, rationale):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO trades(ts, symbol, side, qty, price, fee, rationale) VALUES(?,?,?,?,?,?,?)",
            (ts, symbol, side, qty, price, fee, rationale),
        )
        await db.commit()

async def get_recent_trades(limit=50):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT ts, symbol, side, qty, price, fee, rationale FROM trades ORDER BY id DESC LIMIT ?", (limit,)) as cur:
            rows = await cur.fetchall()
    cols = ["ts","symbol","side","qty","price","fee","rationale"]
    return [dict(zip(cols, r)) for r in rows]
