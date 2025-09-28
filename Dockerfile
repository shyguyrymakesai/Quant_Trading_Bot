FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# system deps (build chain + TA-Lib prerequisite libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# build and install TA-Lib C library from source
RUN wget -q https://downloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib-0.4.0 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd / \
    && rm -rf ta-lib-0.4.0 ta-lib-0.4.0-src.tar.gz \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf \
    && ldconfig

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
