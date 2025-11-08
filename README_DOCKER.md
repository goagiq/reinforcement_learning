# Docker Quick Start

## Prerequisites

- Docker Desktop (Windows) installed and running
- At least 4GB RAM available
- Models directory with trained models (if any)

## Quick Start

### 1. Build and Start

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Access the Application

- **Frontend UI**: http://localhost:3200
- **API Backend**: http://localhost:8200
- **API Docs**: http://localhost:8200/docs
- **NT8 Bridge**: localhost:8888

### 3. Stop the Application

```bash
# Stop services
docker-compose down

# Stop and remove volumes (careful: removes data)
docker-compose down -v
```

## Volume Mappings

The following directories are mapped from your laptop to the container:

- **Models**: `./models` → `/app/models` (your trained models)
- **Data**: `./data` → `/app/data` (training data)
- **NT8 Export**: `C:\Users\sovan\Documents\NinjaTrader 8\export` → `/app/nt8_export` (read-only)
- **Logs**: `./logs` → `/app/logs`

## NT8 Connection

The NT8 Bridge Server runs inside the container on port 8888. Your NT8 strategy should connect to:

```
localhost:8888
```

The container exposes this port to your host machine, so NT8 (running on your laptop) can connect to it.

## Troubleshooting

### Port Already in Use

If port 8200, 3200, or 8888 is already in use:

1. Stop the service using the port, or
2. Edit `docker-compose.yml` to change the host port:
   ```yaml
   ports:
     - "8201:8200"  # Change 8200 to 8201 on host
   ```

### NT8 Cannot Connect

1. Check if the container is running: `docker-compose ps`
2. Check logs: `docker-compose logs nt8-rl-app | grep bridge`
3. Verify port 8888 is exposed: `docker-compose port nt8-rl-app 8888`
4. Test connection: `telnet localhost 8888`

### NT8 Export Directory Not Found

1. Verify the path exists: `dir "C:\Users\sovan\Documents\NinjaTrader 8\export"`
2. Check Docker Desktop file sharing settings
3. Make sure the path in `docker-compose.yml` matches your actual path

### Models Not Found

Ensure the `models` directory exists:
```bash
mkdir models
```

## Advanced Usage

### Run with Monitoring

```bash
docker-compose run --rm nt8-rl-app python start_ui.py --monitoring
```

### Run Custom Commands

```bash
# Run training
docker-compose run --rm nt8-rl-app python src/train.py --config configs/train_config_full.yaml

# Run backtest
docker-compose run --rm nt8-rl-app python src/backtest.py --model models/best_model.pt
```

### View Container Shell

```bash
docker-compose exec nt8-rl-app bash
```

### Rebuild After Code Changes

```bash
docker-compose build --no-cache
docker-compose up -d
```

## For More Details

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for comprehensive documentation.

