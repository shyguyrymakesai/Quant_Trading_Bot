# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BOT_DB_PATH=/app/bot_state.sqlite3

CMD ["python","scripts/run_bot.py"]
