# Multi-stage Dockerfile for NT8 RL Trading System
# Supports both CPU and GPU (CUDA) builds

FROM python:3.13-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/processed data/experience_buffer logs

# Expose ports
# FastAPI backend
EXPOSE 8200
# Frontend dev server (if running in container)
EXPOSE 3200
# NT8 Bridge Server
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "start_ui.py"]

