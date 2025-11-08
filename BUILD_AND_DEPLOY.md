# Build and Deploy Instructions

## Prerequisites Check

Before building, ensure:

1. **Docker Desktop is running**
   - Open Docker Desktop application
   - Wait for it to fully start (whale icon in system tray)
   - Verify it shows "Docker Desktop is running"

2. **Verify Docker is accessible**:
   ```bash
   docker --version
   docker ps
   ```

## Build Steps

### Step 1: Start Docker Desktop

1. Open Docker Desktop from Start Menu
2. Wait for it to fully initialize (may take 1-2 minutes)
3. Check system tray for Docker icon (whale)
4. Verify status shows "Running"

### Step 2: Build the Docker Image

```bash
# Navigate to project directory
cd D:\NT8-RL

# Build the image
docker-compose build

# This may take 10-15 minutes on first build (downloading dependencies)
```

### Step 3: Verify Build Success

```bash
# Check if image was created
docker images | grep nt8-rl

# Should show something like:
# nt8-rl-nt8-rl-app   latest   <image-id>   <size>   <time>
```

### Step 4: Start the Container

```bash
# Start in detached mode
docker-compose up -d

# Or start with logs visible
docker-compose up
```

### Step 5: Verify Services

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Check if services are accessible
curl http://localhost:8200/api/setup/check
```

## Troubleshooting

### Docker Desktop Not Running

**Error**: `error during connect: open //./pipe/dockerDesktopLinuxEngine`

**Solution**:
1. Start Docker Desktop application
2. Wait for it to fully initialize
3. Check system tray for Docker icon
4. Try again: `docker ps`

### Build Fails with "EOF" Error

This can happen if:
- Docker Desktop stops during build
- Network interruption
- Docker daemon restarted

**Solution**:
1. Ensure Docker Desktop is running
2. Retry the build: `docker-compose build`
3. If persists, try: `docker-compose build --no-cache`

### Port Already in Use

**Error**: `Bind for 0.0.0.0:8200 failed: port is already allocated`

**Solution**:
1. Stop existing services using the port
2. Or modify `docker-compose.yml` to use different ports:
   ```yaml
   ports:
     - "8201:8200"  # Change host port
   ```

### Volume Mount Issues (Windows)

**Error**: `invalid mount config for type "bind"`

**Solution**:
1. Verify the NT8 export path exists:
   ```bash
   dir "C:\Users\sovan\Documents\NinjaTrader 8\export"
   ```
2. Enable file sharing in Docker Desktop:
   - Settings → Resources → File Sharing
   - Add `C:\Users\sovan\Documents` if not listed
   - Click "Apply & Restart"
3. Check path format in docker-compose.yml (use forward slashes)

## Quick Start Commands

```bash
# 1. Start Docker Desktop (manual step)

# 2. Build
docker-compose build

# 3. Start
docker-compose up -d

# 4. View logs
docker-compose logs -f nt8-rl-app

# 5. Stop
docker-compose down

# 6. Rebuild after code changes
docker-compose build --no-cache
docker-compose up -d
```

## Access Points

Once running:

- **Frontend**: http://localhost:3200
- **API**: http://localhost:8200
- **API Docs**: http://localhost:8200/docs
- **NT8 Bridge**: localhost:8888

## Next Steps After Deployment

1. **Verify services are running**:
   ```bash
   docker-compose ps
   ```

2. **Check logs for errors**:
   ```bash
   docker-compose logs nt8-rl-app | tail -50
   ```

3. **Test API endpoint**:
   ```bash
   curl http://localhost:8200/api/setup/check
   ```

4. **Access web UI**:
   - Open browser: http://localhost:3200

5. **Configure NT8 Strategy**:
   - Connect to: `localhost:8888`
   - Ensure bridge server is started from the UI

