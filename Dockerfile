FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
# Make pip more resilient to large downloads and retries
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_PREFER_BINARY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/dev/shm/hf \
    TRANSFORMERS_CACHE=/dev/shm/hf

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install hatchling trove-classifiers

# Install the package without build isolation so preinstalled deps are reused
RUN pip install --no-build-isolation .

# Default environment
ENV QQ_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Run the FastAPI app bound to a Unix domain socket
CMD ["uvicorn", "qq.qq_api:app", "--uds", "/run/qq.sock", "--workers", "1"]
