FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

# system deps (TA-Lib via Debian packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ta-lib \
        ta-lib-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
