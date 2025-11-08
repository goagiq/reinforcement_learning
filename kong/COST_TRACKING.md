# Kong Cost Tracking Guide

## Overview

This document describes how to track LLM API costs through Kong Gateway using Prometheus metrics.

## Cost Tracking Setup

### Prometheus Metrics

Kong exposes Prometheus metrics at: `http://localhost:8301/metrics`

### Key Metrics

1. **Request Counts:**
   - `kong_http_requests_total` - Total requests per service
   - `kong_http_requests_total{service="anthropic-service"}` - Anthropic requests
   - `kong_http_requests_total{service="deepseek-service"}` - DeepSeek requests
   - `kong_http_requests_total{service="grok-service"}` - Grok requests
   - `kong_http_requests_total{service="ollama-service"}` - Ollama requests (free)

2. **Request Latency:**
   - `kong_http_request_latency_ms` - Request latency per service

3. **Rate Limit Hits:**
   - `kong_http_requests_total{code="429"}` - Rate limit exceeded

### Provider Pricing (as of 2025)

**Anthropic:**
- Claude Sonnet 4: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- Average request: ~1K tokens input, ~500 tokens output
- Estimated cost: ~$0.003 per request

**DeepSeek Cloud:**
- DeepSeek Chat: ~$0.14 per 1M input tokens, ~$0.28 per 1M output tokens
- Average request: ~1K tokens input, ~500 tokens output
- Estimated cost: ~$0.0002 per request

**Grok (xAI):**
- Grok Beta: ~$0.01 per 1M tokens
- Average request: ~1K tokens
- Estimated cost: ~$0.00001 per request

**Ollama:**
- Local model: $0.00 (no cost)

### Cost Calculation

```python
# Example cost calculation from Prometheus metrics
import requests

def get_service_costs():
    # Fetch Prometheus metrics
    response = requests.get('http://localhost:8301/metrics')
    metrics = response.text
    
    # Parse request counts
    # (Use Prometheus client library for production)
    
    costs = {
        'anthropic': requests * 0.003,
        'deepseek': requests * 0.0002,
        'grok': requests * 0.00001,
        'ollama': 0.0
    }
    
    return costs
```

### Cost Tracking Dashboard

Set up Grafana dashboard with:
- Request counts per service
- Estimated costs per service
- Daily/monthly cost projections
- Cost per consumer

### Alerting

Set up alerts for:
- Cost threshold exceeded (e.g., $100/day)
- Unusual request spikes
- Rate limit hits > 10%

## Monitoring Commands

```bash
# View Prometheus metrics
curl http://localhost:8301/metrics

# Filter by service
curl http://localhost:8301/metrics | grep "anthropic-service"

# Get request counts
curl http://localhost:8301/metrics | grep "kong_http_requests_total"
```

## Cost Optimization

1. **Use Ollama for development** (free)
2. **Cache responses** (Phase 4)
3. **Monitor and optimize** rate limits
4. **Set cost budgets** per consumer
5. **Use cheaper providers** when possible
