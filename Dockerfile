# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies required by pandas/numpy/psutil/etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libatlas-base-dev \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        libstdc++6 \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install psutil

COPY . .

# Install the project as a package so entry points resolve consistently.
RUN pip install .

ENTRYPOINT ["python", "-m", "cryptopy.scripts.trading.run_triangular_arbitrage"]
CMD ["--help"]
