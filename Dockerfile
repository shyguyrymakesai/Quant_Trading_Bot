############################
# 1) Builder: compile TA-Lib C and wheels
############################
FROM python:3.11-slim AS wheels
ARG TA_LIB_VERSION=0.4.0
WORKDIR /build

# toolchain + curl for TA-Lib C lib + building Python wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool \
      python3-dev ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/*

# --- Download TA-Lib C from GitHub Releases (reliable mirror) ---
# (avoid SourceForge redirect/HTML issues)
RUN set -euxo pipefail; \
    curl -fL --retry 5 --retry-connrefused \
      -o /tmp/ta-lib-src.tgz \
      "https://github.com/TA-Lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    mkdir -p /tmp/ta-lib; \
    tar -xzf /tmp/ta-lib-src.tgz -C /tmp/ta-lib --strip-components=1; \
    cd /tmp/ta-lib; \
    ./configure --prefix=/usr/local; \
    make -j"$(nproc)"; \
    make install; \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf; \
    ldconfig; \
    rm -rf /tmp/ta-lib /tmp/ta-lib-src.tgz

# Preinstall helpers + numpy (TA-Lib Python wrapper needs numpy headers)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir numpy

# Cache-friendly: only copy requirements first
COPY requirements.txt /build/requirements.txt

# Build wheels for ALL deps (including TA-Lib if present)
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r /build/requirements.txt

############################
# 2) Final runtime image
############################
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

# TA-Lib C runtime from builder
COPY --from=wheels /usr/local/lib/libta_lib.* /usr/local/lib/
COPY --from=wheels /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/ta-lib.conf && ldconfig

# Install wheels (no compiling in final image)
COPY --from=wheels /build/wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt \
 && rm -rf /wheels

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/run_once.py"]
