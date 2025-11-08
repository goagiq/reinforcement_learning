#!/bin/bash
# Docker entrypoint script for NT8 RL Trading System

set -e

echo "=========================================="
echo "NT8 RL Trading System - Container Startup"
echo "=========================================="
echo ""

# Check if models directory exists and is writable
if [ ! -d "/app/models" ]; then
    echo "Creating models directory..."
    mkdir -p /app/models
fi

# Check if data directories exist
mkdir -p /app/data/raw /app/data/processed /app/data/experience_buffer
mkdir -p /app/logs

# Check NT8 export directory access
if [ -d "/app/nt8_export" ]; then
    echo "✓ NT8 export directory accessible: /app/nt8_export"
    ls -la /app/nt8_export | head -5
else
    echo "⚠ Warning: NT8 export directory not accessible: /app/nt8_export"
    echo "  Make sure the volume is properly mounted"
fi

# Set environment variables for bridge server
export NT8_BRIDGE_HOST=${NT8_BRIDGE_HOST:-"0.0.0.0"}
export NT8_BRIDGE_PORT=${NT8_BRIDGE_PORT:-8888}

echo ""
echo "Environment Configuration:"
echo "  NT8_BRIDGE_HOST: $NT8_BRIDGE_HOST"
echo "  NT8_BRIDGE_PORT: $NT8_BRIDGE_PORT"
echo "  DATA_DIR: ${DATA_DIR:-/app/data}"
echo "  MODELS_DIR: ${MODELS_DIR:-/app/models}"
echo "  NT8_EXPORT_DIR: ${NT8_EXPORT_DIR:-/app/nt8_export}"
echo ""

# Check if we're starting with start_ui.py or another command
if [ "$#" -eq 0 ] || [ "$1" = "start_ui.py" ] || [ "$1" = "python" ] && [ "$2" = "start_ui.py" ]; then
    echo "Starting NT8 RL Trading System UI..."
    echo ""
    
    # Check if --monitoring flag is present
    if echo "$@" | grep -q "\--monitoring"; then
        echo "Monitoring services will be started via start_ui.py"
    fi
    
    # Execute start_ui.py with all arguments
    exec python start_ui.py "$@"
else
    # Execute custom command
    exec "$@"
fi

