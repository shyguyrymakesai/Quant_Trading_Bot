# Use Ubuntu-based Python where TA-Lib is packaged
FROM python:3.11-jammy

ENV PYTHONUNBUFFERED=1

# System deps: TA-Lib C lib + headers, compiler (for any other wheels), tzdata
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ta-lib ta-lib-dev libta-lib0 \
      build-essential \
      tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
# 1) Upgrade pip tooling
# 2) Install numpy first (TA-Lib wrapper needs it)
# 3) Install requirements (including TA-Lib Python wrapper)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir numpy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
