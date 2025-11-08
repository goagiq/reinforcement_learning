# Kong Port Configuration

## Port Assignment

Kong Gateway uses the following ports:

| Service | Port | Purpose |
|---------|------|---------|
| Kong Proxy | 8300 | Main gateway endpoint (routes requests) |
| Kong Admin API | 8301 | Administrative API (configuration) |
| Kong PostgreSQL | 5434 | Database for Kong (avoids conflicts) |

## Port Conflicts Avoided

### Why Port 5434 for PostgreSQL?

- **Port 5432**: Used by your existing PostgreSQL instance (`docker-postgres-1`)
- **Port 5433**: Reserved for pgvector (your existing setup)
- **Port 5434**: Used by Kong PostgreSQL ✅

This ensures no conflicts with:
- Existing PostgreSQL on port 5432
- pgvector on port 5433
- Kong PostgreSQL on port 5434

## Changing Ports

If you need to change Kong ports:

### Change Proxy/Admin Ports:

Edit `kong/docker-compose.yml`:
```yaml
ports:
  - "NEW_PROXY_PORT:8300"   # Change external port
  - "NEW_ADMIN_PORT:8301"    # Change external port

environment:
  KONG_PROXY_LISTEN: 0.0.0.0:8300  # Internal port (keep as-is)
  KONG_ADMIN_LISTEN: 0.0.0.0:8301   # Internal port (keep as-is)
```

### Change PostgreSQL Port:

Edit `kong/docker-compose.yml`:
```yaml
postgres:
  ports:
    - "NEW_PORT:5432"  # Change external port (internal stays 5432)
```

After changes:
```bash
cd kong
docker-compose down
docker-compose up -d
```

## Port Verification

Check current port usage:
```bash
# Check Kong ports
docker ps | grep kong

# Check PostgreSQL ports
docker ps | grep postgres

# Check if ports are in use
netstat -ano | findstr "8300 8301 5434"  # Windows
ss -tuln | grep -E "8300|8301|5434"      # Linux
```

## Current Configuration

- ✅ Kong Proxy: `http://localhost:8300`
- ✅ Kong Admin: `http://localhost:8301`
- ✅ Kong PostgreSQL: `localhost:5434`
- ✅ No conflicts with existing services

