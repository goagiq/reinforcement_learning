# FastAPI Health Endpoint for Kong Health Checks

## Overview

Kong health checks require a health endpoint on your FastAPI service. This guide shows how to add one.

## Quick Implementation

Add this to your FastAPI app (`src/api_server.py`):

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for Kong"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "nt8-rl-api",
            "timestamp": datetime.now().isoformat()
        }
    )
```

## Enhanced Health Check

For more detailed health information:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime
import psutil
import os

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Check critical dependencies
        checks = {
            "api": "healthy",
            "database": "healthy",  # Add your DB check
            "model": "loaded" if model_loaded else "not_loaded",
        }
        
        # Determine overall health
        all_healthy = all(v in ["healthy", "loaded"] for v in checks.values())
        status_code = 200 if all_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if all_healthy else "degraded",
                "service": "nt8-rl-api",
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024)
                }
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
```

## Kong Health Check Configuration

Kong is configured to:
- Check `/health` endpoint every 10 seconds
- Mark unhealthy after 3 consecutive failures
- Mark healthy after 3 consecutive successes

## Testing

```bash
# Test health endpoint
curl http://localhost:8200/health

# Test through Kong
curl http://localhost:8300/api/health
```

## Response Codes

- **200:** Service is healthy
- **503:** Service is unhealthy (Kong will stop routing)

---

**Note:** Make sure your FastAPI service has a `/health` endpoint for Kong health checks to work properly.
