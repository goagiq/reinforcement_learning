# Kong Monitoring API Guide

## Overview

This document describes how to access and use Kong's monitoring and observability features.

## Metrics Endpoint

### Prometheus Metrics

**Endpoint:** `http://localhost:8301/metrics`

**Access:** No authentication required (admin API)

**Example:**
```bash
curl http://localhost:8301/metrics
```

### Key Metrics

1. **Request Metrics:**
   - `kong_http_requests_total` - Total HTTP requests
   - `kong_http_requests_latency_ms` - Request latency
   - `kong_http_requests_consumer_total` - Requests per consumer

2. **Service Metrics:**
   - `kong_http_requests_total{service="anthropic-service"}` - Requests per service
   - `kong_http_requests_status{service="anthropic-service",code="200"}` - Status codes per service

3. **Rate Limiting:**
   - `kong_ratelimiting_usage{service="anthropic-service"}` - Rate limit usage

4. **Cache Metrics:**
   - `kong_proxy_cache_total{service="anthropic-service",status="hit"}` - Cache hits
   - `kong_proxy_cache_total{service="anthropic-service",status="miss"}` - Cache misses

5. **Upstream Metrics:**
   - `kong_upstream_latency_ms` - Upstream latency
   - `kong_upstream_healthchecks_total` - Health check results

## Service Health

**Endpoint:** `http://localhost:8301/services/{service-name}/health`

**Example:**
```bash
curl http://localhost:8301/services/fastapi-service/health
```

## Service Statistics

**Endpoint:** `http://localhost:8301/services/{service-name}/stats`

**Example:**
```bash
curl http://localhost:8301/services/anthropic-service/stats
```

## Consumer Statistics

**Endpoint:** `http://localhost:8301/consumers/{consumer-name}/stats`

**Example:**
```bash
curl http://localhost:8301/consumers/reasoning-engine-consumer/stats
```

## Logging

Kong logs are available via:
- Docker logs: `docker logs kong-gateway`
- HTTP log plugin: Configured per service
- File log plugin: If configured

## Monitoring Queries

### Get Total Requests
```bash
curl http://localhost:8301/metrics | grep "kong_http_requests_total" | grep -v "#"
```

### Get Requests by Service
```bash
curl http://localhost:8301/metrics | grep "kong_http_requests_total.*service"
```

### Get Error Rate
```bash
curl http://localhost:8301/metrics | grep "kong_http_requests_status.*code=\"5"
```

### Get Cache Hit Rate
```bash
curl http://localhost:8301/metrics | grep "kong_proxy_cache_total.*status=\"hit\""
```

## Integration with Prometheus

1. **Configure Prometheus:**
   - Use `kong/prometheus.yml` configuration
   - Scrape interval: 10-15 seconds
   - Target: `localhost:8301`

2. **Prometheus Queries:**
   ```promql
   # Total requests per service
   sum(rate(kong_http_requests_total[5m])) by (service)
   
   # Error rate
   sum(rate(kong_http_requests_status{code=~"5.."}[5m])) / sum(rate(kong_http_requests_total[5m]))
   
   # Cache hit rate
   sum(rate(kong_proxy_cache_total{status="hit"}[5m])) / sum(rate(kong_proxy_cache_total[5m]))
   ```

## Integration with Grafana

See `kong/grafana-dashboard.json` for Grafana dashboard configuration.

## Alerts

Alert configurations are in `kong/alerts.json`. Integrate with:
- Prometheus Alertmanager
- Grafana alerts
- Custom alerting systems

## FastAPI Monitoring Endpoints

FastAPI provides monitoring endpoints that aggregate Kong metrics:

- `/api/monitoring/health` - Kong health status
- `/api/monitoring/metrics` - Kong metrics summary
- `/api/monitoring/services` - Kong services status

**Note:** These endpoints require FastAPI server restart to be available.

