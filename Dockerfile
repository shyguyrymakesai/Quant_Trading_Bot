############################
# 1) Builder: TA-Lib (C)
############################
FROM python:3.11-slim AS talib-builder
ARG TA_LIB_VERSION=0.4.0
WORKDIR /tmp

# Toolchain for building the C library
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      autoconf automake libtool \
      ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/*

# Download (follow redirects), verify, build, install into /usr/local
RUN set -euxo pipefail; \
    curl -fsSL "https://sourceforge.net/projects/ta-lib/files/ta-lib/${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz/download" \
      -o ta-lib-src.tgz; \
    # quick integrity check: ensure it's a gzip, not HTML
    file ta-lib-src.tgz | grep -qi 'gzip compressed data'; \
    mkdir -p /tmp/ta-lib; \
    tar -xzf ta-lib-src.tgz -C /tmp/ta-lib; \
    cd /tmp/ta-lib/ta-lib-${TA_LIB_VERSION}; \
    ./configure --prefix=/usr/local; \
    make -j"$(nproc)"; \
    make install; \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf; \
    ldconfig

############################
# 2) Final runtime image
############################
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Minimal runtime libs + timezone
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
 && rm -rf /var/lib/apt/lists/*

# Copy only the built TA-Lib artifacts from builder
COPY --from=talib-builder /usr/local/lib/libta_lib.* /usr/local/lib/
COPY --from=talib-builder /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf && ldconfig

# Python deps (numpy first helps TA-Lib wheel build)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir numpy

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
