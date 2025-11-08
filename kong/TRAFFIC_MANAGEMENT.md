# Kong Traffic Management Guide

## Overview

This document describes the traffic management features enabled in Kong Gateway for Phase 4, including caching, retry logic, health checks, and load balancing.

## Features Enabled

### 1. Proxy Caching

**Status:** ✅ Enabled on all LLM services

**Configuration:**
- **Cache TTL:** 300 seconds (5 minutes)
- **Storage TTL:** 600 seconds (10 minutes)
- **Cache Method:** Memory-based (in-memory dictionary)
- **Cached Responses:** HTTP 200, GET and POST requests
- **Content Type:** application/json

**Benefits:**
- Reduces API costs by caching repeated requests
- Improves response times for cached responses
- Reduces load on LLM providers

**How It Works:**
- Kong caches responses based on request URL and headers
- Cache key includes: method, path, query parameters
- Cache is invalidated after TTL expires

**Example:**
```bash
# First request - hits LLM API
curl -H "apikey: YOUR_KEY" http://localhost:8300/llm/anthropic/v1/messages

# Second identical request - served from cache (faster, no API cost)
curl -H "apikey: YOUR_KEY" http://localhost:8300/llm/anthropic/v1/messages
```

**Cache Headers:**
- `X-Cache-Status: HIT` - Response served from cache
- `X-Cache-Status: MISS` - Response fetched from upstream
- `X-Cache-Key: <cache-key>` - Cache key used

### 2. Retry Logic

**Status:** ✅ Enabled on all services

**Configuration:**
- **Retries:** 3 attempts
- **Retry On:**
  - HTTP 5xx errors (500, 502, 503, 504)
  - Rate limit errors (429)
  - Timeouts
- **Methods:** GET, POST, PUT, DELETE
- **Timeout:** 1000ms between retries

**Benefits:**
- Improves reliability during transient failures
- Handles rate limit errors gracefully
- Reduces impact of network issues

**How It Works:**
1. Request fails with 5xx or timeout
2. Kong automatically retries up to 3 times
3. If all retries fail, returns error to client
4. Exponential backoff between retries (built into Kong)

**Example:**
```bash
# If upstream returns 503, Kong will retry 3 times before returning error
curl -H "apikey: YOUR_KEY" http://localhost:8300/llm/anthropic/v1/messages
```

### 3. Health Checks

**Status:** ✅ Enabled on local services (Ollama, FastAPI)

**Configuration:**

#### Ollama Service
- **Type:** HTTP health check
- **Path:** `/api/tags`
- **Interval:** 10 seconds
- **Timeout:** 5 seconds
- **Healthy Threshold:** 3 consecutive successes
- **Unhealthy Threshold:** 3 consecutive failures

#### FastAPI Service
- **Type:** HTTP health check
- **Path:** `/health` (must exist in FastAPI app)
- **Interval:** 10 seconds
- **Timeout:** 5 seconds
- **Healthy Threshold:** 3 consecutive successes
- **Unhealthy Threshold:** 3 consecutive failures

**Passive Health Checks:**
- **Healthy Statuses:** 200, 201, 202, 204, 301, 302, 307, 308
- **Unhealthy Statuses:** 429, 500, 502, 503, 504
- **Unhealthy Timeouts:** 5 seconds

**Benefits:**
- Automatically detects unhealthy services
- Prevents routing to failed services
- Improves system reliability

**How It Works:**
1. Kong periodically checks service health
2. If service fails 3 consecutive checks, marked unhealthy
3. Kong stops routing to unhealthy services
4. When service recovers, marked healthy again

### 4. Load Balancing (Future)

**Status:** ⏳ Not yet configured (single instances)

**Future Configuration:**
- Multiple upstream targets per service
- Load balancing algorithms:
  - Round-robin (default)
  - Least connections
  - IP hash (for session affinity)

**Example Configuration:**
```yaml
upstreams:
  - name: ollama-upstream
    targets:
      - target: host.docker.internal:11434
        weight: 100
      - target: ollama-replica:11434
        weight: 100
    algorithm: round-robin
```

### 5. Intelligent Routing (Future)

**Status:** ⏳ Not yet configured

**Future Configuration:**
- Route based on cost (prefer cheaper providers)
- Route based on latency (prefer faster providers)
- Route based on availability (prefer healthy providers)
- Fallback chains (try primary, then secondary)

**Example:**
```
Request → Try Anthropic (cost: $0.003)
  ↓ (if fails)
Try DeepSeek (cost: $0.0002)
  ↓ (if fails)
Try Ollama (cost: $0.00)
```

## Monitoring

### Check Cache Status
```bash
# Make a request and check cache headers
curl -v -H "apikey: YOUR_KEY" http://localhost:8300/llm/anthropic/v1/messages \
  | grep -i "x-cache"
```

### Check Service Health
```bash
# Check Ollama health
curl http://localhost:8301/services/ollama-service/health

# Check FastAPI health
curl http://localhost:8301/services/fastapi-service/health
```

### View Retry Metrics
```bash
# Check Prometheus metrics for retry attempts
curl http://localhost:8301/metrics | grep -i retry
```

## Configuration Details

### Cache Plugin Configuration
- **Memory Cache:** Uses Kong's shared dictionary
- **TTL Strategy:** Per-request TTL (configurable)
- **Cache Control:** Respects upstream Cache-Control headers

### Retry Plugin Configuration
- **Backoff:** Exponential backoff (handled by Kong)
- **Retry Conditions:** Configurable per service
- **Timeout:** Configurable per retry attempt

### Health Check Configuration
- **Active Checks:** Periodic health probes
- **Passive Checks:** Monitor actual request responses
- **Circuit Breaker:** Automatic failover on unhealthy state

## Best Practices

1. **Cache TTL:** Set appropriate TTL based on data freshness requirements
   - LLM responses: 5 minutes (good balance)
   - Static content: Longer TTL
   - Dynamic content: Shorter TTL

2. **Retry Strategy:** Don't retry on 4xx errors (client errors)
   - Only retry on 5xx (server errors) and timeouts
   - Avoid retrying on 401 (authentication) errors

3. **Health Checks:** Use lightweight endpoints
   - `/health` or `/ping` endpoints
   - Avoid expensive operations in health checks

4. **Monitoring:** Track cache hit rates and retry counts
   - High cache hit rate = good caching strategy
   - High retry count = upstream reliability issues

## Troubleshooting

### Cache Not Working
- Check if `proxy-cache` plugin is enabled
- Verify cache TTL hasn't expired
- Check cache headers in response

### Retries Not Working
- Verify `retry` plugin is enabled
- Check retry configuration (HTTP statuses, methods)
- Review Kong logs for retry attempts

### Health Checks Failing
- Verify health check endpoint exists
- Check service is actually healthy
- Review health check configuration

## Performance Impact

### Caching
- **Memory Usage:** ~100MB per 1000 cached responses
- **CPU Impact:** Minimal (dictionary lookup)
- **Latency Reduction:** 90-99% for cached responses

### Retry Logic
- **Latency Impact:** +1-3 seconds per retry
- **CPU Impact:** Minimal
- **Network Impact:** Additional requests to upstream

### Health Checks
- **Network Impact:** 1 request per 10 seconds per service
- **CPU Impact:** Minimal
- **Latency Impact:** None (async checks)

---

**Phase 4 Status:** ✅ Complete
**Last Updated:** November 4, 2025
