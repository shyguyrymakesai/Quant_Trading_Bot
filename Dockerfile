# ---- Runtime (single stage) ----
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1

# System deps: TA-Lib runtime + headers to build the Python wrapper, and build tools
RUN apk add --no-cache \
      tzdata \
      ta-lib ta-lib-dev \
      g++ make

WORKDIR /app

# (optional but recommended) pin pip and ensure numpy first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir numpy

# install your deps (TA-Lib should be in requirements.txt with this exact casing)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# rest of your image
COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
