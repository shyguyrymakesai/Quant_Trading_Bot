# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# (optional) build deps if your libs need it
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# install minimal deps for cloud stub
COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY src/ ./src/

# run one cycle then exit (cheap Fargate runs)
ENTRYPOINT ["python", "-m", "src.run_once"]
