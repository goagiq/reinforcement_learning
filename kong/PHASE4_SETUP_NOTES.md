# Phase 4 Setup Notes

## Current Status

Phase 4 (Traffic Management) requires Kong restart to enable plugins and configure shared dictionaries.

## Required Changes

### 1. Update docker-compose.yml

Added:
- `KONG_PLUGINS: bundled,proxy-cache,retry` - Enable plugins
- `KONG_NGINX_HTTP_LUA_SHARED_DICT: kong_cache 128m` - Shared dictionary for proxy-cache

### 2. Restart Kong

After updating docker-compose.yml:

```bash
cd kong
docker-compose down
docker-compose up -d
```

### 3. Run Setup Script

After restart:

```bash
cd kong
bash setup_phase4_fixed.sh
```

## What Will Be Configured

1. **Proxy Cache Plugin**
   - Enabled on: anthropic-service, deepseek-service, grok-service, ollama-service
   - Cache TTL: 300 seconds (5 minutes)
   - Storage TTL: 600 seconds (10 minutes)
   - Strategy: Memory-based

2. **Retry Plugin**
   - Enabled on: All services
   - Retries: 3 attempts
   - Retry on: 5xx errors, timeouts, rate limits (429)

3. **Health Checks**
   - Ollama: `/api/tags` endpoint
   - FastAPI: `/health` endpoint (needs to be added to FastAPI app)

## Documentation Created

- `kong/TRAFFIC_MANAGEMENT.md` - Complete traffic management guide
- `kong/FASTAPI_HEALTH_ENDPOINT.md` - Guide for adding health endpoint

## Next Steps After Restart

1. Run `setup_phase4_fixed.sh` to enable plugins
2. Add `/health` endpoint to FastAPI (see `FASTAPI_HEALTH_ENDPOINT.md`)
3. Test caching with repeated requests
4. Test retry logic with error simulation

