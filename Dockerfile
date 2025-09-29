FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# ---- system build deps (for TA-Lib + Python C extension) ----
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        autoconf \
        automake \
        libtool \
        wget \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# ---- build & install TA-Lib C library (0.4.0) ----
ARG TA_LIB_VERSION=0.4.0
RUN set -eux; \
    wget -q "https://downloads.sourceforge.net/ta-lib/ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    tar -xzf "ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    cd "ta-lib-${TA_LIB_VERSION}"; \
    ./configure --prefix=/usr/local; \
    make -j"$(nproc)"; \
    make install; \
    cd /; \
    rm -rf "ta-lib-${TA_LIB_VERSION}" "ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf; \
    ldconfig

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
