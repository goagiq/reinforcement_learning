# Docker Setup Guide for NT8 RL Trading System

This guide explains how to containerize and run the NT8 RL Trading System using Docker.

## Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux/Mac)
- Docker Compose v2.0+
- At least 4GB RAM available for containers
- Models directory with trained models (if any)

## Quick Start

### 1. Build and Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Access Services

- **FastAPI Backend**: http://localhost:8200
- **Frontend UI**: http://localhost:3200
- **NT8 Bridge Server**: localhost:8888
- **API Documentation**: http://localhost:8200/docs

## Configuration

### Volume Mappings

The Docker setup maps the following directories:

1. **Models Directory** (`./models` → `/app/models`)
   - Stores trained models
   - Persists across container restarts

2. **Data Directory** (`./data` → `/app/data`)
   - Raw data: `data/raw/`
   - Processed data: `data/processed/`
   - Experience buffer: `data/experience_buffer/`

3. **NT8 Export Directory** (`C:\Users\sovan\Documents\NinjaTrader 8\export` → `/app/nt8_export`)
   - Read-only access to NT8 export directory
   - New training files are automatically available

4. **Logs Directory** (`./logs` → `/app/logs`)
   - Application logs

### Port Mappings

- **8200**: FastAPI backend
- **3200**: Frontend development server
- **8888**: NT8 Bridge Server (for NT8 strategy connection)

### Environment Variables

Key environment variables:

- `NT8_BRIDGE_HOST`: Bridge server host (default: `0.0.0.0`)
- `NT8_BRIDGE_PORT`: Bridge server port (default: `8888`)
- `DATA_DIR`: Data directory path (default: `/app/data`)
- `MODELS_DIR`: Models directory path (default: `/app/models`)
- `NT8_EXPORT_DIR`: NT8 export directory (default: `/app/nt8_export`)

## NT8 Bridge Server Connection

### Option 1: Bridge Server in Container (Recommended)

The bridge server runs inside the container and accepts connections from NT8 on the host:

1. **Container Configuration**:
   - Bridge server binds to `0.0.0.0:8888` inside container
   - Port `8888` is exposed to host

2. **NT8 Strategy Configuration**:
   - Connect to: `localhost:8888`
   - The container's port is mapped to the host

### Option 2: Bridge Server on Host

If the bridge server runs separately on the host:

1. **Container Configuration**:
   - Set `NT8_BRIDGE_HOST=host.docker.internal`
   - Set `NT8_BRIDGE_PORT=8888`

2. **Host Configuration**:
   - Bridge server must bind to `0.0.0.0:8888` (not `localhost`)
   - Or use host network mode: `network_mode: "host"`

## Windows Path Mappings

For Windows, the NT8 export path is mapped as:

```yaml
volumes:
  - "C:/Users/sovan/Documents/NinjaTrader 8/export:/app/nt8_export:ro"
```

**Note**: 
- Use forward slashes `/` or escaped backslashes `\\` in docker-compose.yml
- The path is case-sensitive on Linux containers

## Running with Monitoring

To start with Prometheus and Grafana:

```bash
docker-compose run --rm nt8-rl-app python start_ui.py --monitoring
```

Or start Kong and monitoring separately:

```bash
# Start Kong Gateway
cd kong && docker-compose up -d

# Start monitoring
cd kong && docker-compose -f docker-compose-prometheus.yml up -d

# Start main application
docker-compose up -d
```

## GPU Support

For GPU support (NVIDIA):

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Uncomment GPU configuration in `docker-compose.override.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Set environment variable:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

## Custom Configuration

Create `docker-compose.override.yml` to override default settings:

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
# Edit docker-compose.override.yml with your settings
```

## Troubleshooting

### Port Already in Use

If ports are already in use:

```bash
# Check what's using the port
netstat -ano | findstr :8200  # Windows
lsof -i :8200                 # Linux/Mac

# Change ports in docker-compose.yml
ports:
  - "8201:8200"  # Change host port
```

### Volume Mount Issues (Windows)

If volume mounts fail:

1. Enable file sharing in Docker Desktop
2. Use WSL2 backend (recommended)
3. Check path format (use `/` or `\\`)

### NT8 Cannot Connect to Bridge Server

1. **Check bridge server is running**:
   ```bash
   docker-compose logs nt8-rl-app | grep bridge
   ```

2. **Verify port mapping**:
   ```bash
   docker-compose ps
   ```

3. **Test connection**:
   ```bash
   telnet localhost 8888
   ```

4. **Check firewall**: Ensure port 8888 is not blocked

### Models Not Found

Ensure models directory exists and is mounted:

```bash
# Check volume mount
docker-compose exec nt8-rl-app ls -la /app/models

# Create models directory if missing
mkdir -p models
```

### NT8 Export Directory Not Accessible

1. **Check path in docker-compose.yml**:
   - Windows: `C:/Users/sovan/Documents/NinjaTrader 8/export`
   - Use forward slashes or escaped backslashes

2. **Verify directory exists**:
   ```bash
   dir "C:\Users\sovan\Documents\NinjaTrader 8\export"
   ```

3. **Check Docker Desktop file sharing**:
   - Settings → Resources → File Sharing
   - Add `C:\Users\sovan\Documents` if not listed

## Development Mode

For development with hot reload:

```bash
# Mount source code as volume
volumes:
  - .:/app
  - /app/node_modules  # Exclude node_modules
```

## Production Deployment

For production:

1. Use multi-stage builds
2. Set resource limits
3. Use Docker secrets for sensitive data
4. Enable health checks
5. Set up logging aggregation
6. Use Docker Swarm or Kubernetes for orchestration

## Health Checks

Add health checks to docker-compose.yml:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8200/api/setup/check"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Backup and Restore

### Backup Models and Data

```bash
# Backup models
docker-compose exec nt8-rl-app tar -czf /tmp/models-backup.tar.gz /app/models
docker cp nt8-rl-app:/tmp/models-backup.tar.gz ./models-backup.tar.gz

# Backup data
docker-compose exec nt8-rl-app tar -czf /tmp/data-backup.tar.gz /app/data
docker cp nt8-rl-app:/tmp/data-backup.tar.gz ./data-backup.tar.gz
```

### Restore

```bash
# Restore models
docker cp ./models-backup.tar.gz nt8-rl-app:/tmp/
docker-compose exec nt8-rl-app tar -xzf /tmp/models-backup.tar.gz -C /app

# Restore data
docker cp ./data-backup.tar.gz nt8-rl-app:/tmp/
docker-compose exec nt8-rl-app tar -xzf /tmp/data-backup.tar.gz -C /app
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NT8 Bridge Server Guide](docs/BRIDGE_SERVER_EXPLAINED.md)
- [Kong Gateway Setup](kong/README.md)

