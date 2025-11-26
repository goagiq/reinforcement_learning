#!/bin/bash
# Start script for NT8 RL Trading System Backend

# Enable unbuffered Python output (ensures Priority 1 messages appear immediately)
export PYTHONUNBUFFERED=1

# Set Python encoding (helps with special characters)
export PYTHONIOENCODING=utf-8

# Start backend API server
echo "Starting backend API server..."
echo "  PYTHONUNBUFFERED=1 (unbuffered output enabled)"
echo "  Priority 1 messages will appear immediately"
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv run to start backend..."
    uv run python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
else
    echo "Using system Python to start backend..."
    python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
fi

