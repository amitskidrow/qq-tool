FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
# Make pip more resilient to large downloads and retries
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for scientific stack (optional but helps wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md /app/
COPY src /app/src

# Upgrade pip tooling and preinstall build backend to avoid resolver issues
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install hatchling trove-classifiers

# Install a ROCm-enabled PyTorch build for AMD GPUs.
# Allow overriding the ROCm index at build time for host compatibility.
# Common options: rocm6.0, rocm6.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.0
RUN pip install --index-url ${TORCH_INDEX_URL} \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install the package without build isolation so preinstalled deps are reused
RUN pip install --no-build-isolation .

# Expose API port
EXPOSE 8787

# Default environment (can be overridden via compose)
ENV QQ_TYPESENSE_HOST=localhost \
    QQ_TYPESENSE_PORT=8108 \
    QQ_TYPESENSE_PROTOCOL=http \
    QQ_TYPESENSE_API_KEY=tsdev \
    QQ_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    QQ_EMBED_DEVICE=cuda

# Run the FastAPI app
CMD ["uvicorn", "qq.api:app", "--host", "0.0.0.0", "--port", "8787", "--workers", "1"]
