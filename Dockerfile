# =============================================================================
# Sift-1B — Containerized Inference Service
# =============================================================================
# Packages the GGUF model + FastAPI server into a single Docker container.
#
# Build:
#   docker build -t sift-1b .
#
# Run (CPU):
#   docker run -p 8000:8000 sift-1b
#
# Run (GPU):
#   docker run -p 8000:8000 --gpus all sift-1b
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    llama-cpp-python \
    pydantic

# Copy server code
COPY serve/server.py /app/server.py

# Copy the GGUF model (build after export)
# NOTE: You must have run export/export_gguf.py first
COPY export/sift-1b/ /app/models/

# Expose the inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["python", "server.py", \
     "--model", "/app/models/unsloth.Q4_K_M.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
