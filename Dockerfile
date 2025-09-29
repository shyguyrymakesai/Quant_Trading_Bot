# Use Ubuntu 22.04 with Python 3.11
FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and system deps including TA-Lib
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv \
      python3-pip \
      ta-lib libta-lib-dev libta-lib0 \
      build-essential \
      tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Python deps
# 1) Upgrade pip tooling
# 2) Install numpy first (TA-Lib wrapper needs it)
# 3) Install requirements (including TA-Lib Python wrapper)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir numpy

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
