FROM python:3.12-slim

# Minimal system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

# Faster, cleaner pip
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --upgrade pip

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY . .

# Default command
CMD ["python", "scripts/run_bot.py"]
