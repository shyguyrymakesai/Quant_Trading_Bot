# Use Ubuntu 22.04 base and install Python 3.11
FROM ubuntu:jammy

ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and TA-Lib C library + headers from apt
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv \
      python3-pip \
      libta-lib0 ta-lib-dev \
      build-essential tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Python deps (numpy first, then wrapper)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir numpy \
 && python -m pip install --no-cache-dir TA-Lib

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
