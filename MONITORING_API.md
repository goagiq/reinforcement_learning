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

## Service Health

**Endpoint:** `http://localhost:8301/services/{service-name}/health`

**Example:**
```bash
curl http://localhost:8301/services/fastapi-service/health
```

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

## Integration with Prometheus

Use `kong/prometheus.yml` configuration file.
