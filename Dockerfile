FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Skip TA-Lib - using pandas-ta for pure Python indicators
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
