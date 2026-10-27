FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libhdf5-dev nlohmann-json3-dev \
    libxrender1 libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fredis
COPY . .
RUN python scripts/install.py \
    && python -c "import msentity, msentity_similarity"

WORKDIR /data
CMD ["python", "-m", "fredis.cli", "--help"]
