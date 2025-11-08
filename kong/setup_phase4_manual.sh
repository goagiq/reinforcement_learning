#!/bin/bash
# Phase 4: Traffic Management Setup (Manual Plugin Creation)

KONG_ADMIN="http://localhost:8301"

echo "🔧 Phase 4: Manually Setting up Traffic Management Plugins..."
echo "=============================================================="
echo ""

# 1. Enable Proxy Cache Plugin (JSON format)
echo "1️⃣  Enabling Proxy Cache Plugin..."

for service in anthropic-service deepseek-service grok-service ollama-service; do
    echo "   Configuring cache for $service..."
    
    RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "proxy-cache",
        "config": {
          "response_code": [200],
          "request_method": ["GET", "POST"],
          "content_type": ["application/json"],
          "cache_ttl": 300,
          "storage_ttl": 600,
          "memory": {
            "dictionary_name": "kong_cache"
          }
        }
      }')
    
    PLUGIN_ID=$(echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null || python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null)
    
    if [ -n "$PLUGIN_ID" ] && [ "$PLUGIN_ID" != "error" ]; then
        echo "   ✅ Proxy cache enabled for $service (ID: $PLUGIN_ID)"
    else
        # Check if it already exists
        EXISTING=$(curl -s "$KONG_ADMIN/services/$service/plugins" | python -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'proxy-cache']
print('exists' if plugins else '')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'proxy-cache']
print('exists' if plugins else '')
" 2>/dev/null)
        
        if [ "$EXISTING" = "exists" ]; then
            echo "   ⚠️  Proxy cache already exists for $service"
        else
            echo "   ❌ Failed to create proxy cache for $service"
            echo "      Response: $(echo "$RESPONSE" | head -3)"
        fi
    fi
done

echo "✅ Proxy caching configured"

# 2. Enable Retry Plugin (JSON format)
echo ""
echo "2️⃣  Enabling Retry Plugin..."

for service in anthropic-service deepseek-service grok-service ollama-service fastapi-service; do
    echo "   Configuring retry for $service..."
    
    RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "retry",
        "config": {
          "retries": 3,
          "methods": ["GET", "POST", "PUT", "DELETE"],
          "http_statuses": [500, 502, 503, 504, 429],
          "timeout": 1000
        }
      }')
    
    PLUGIN_ID=$(echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null || python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null)
    
    if [ -n "$PLUGIN_ID" ] && [ "$PLUGIN_ID" != "error" ]; then
        echo "   ✅ Retry enabled for $service (ID: $PLUGIN_ID)"
    else
        # Check if it already exists
        EXISTING=$(curl -s "$KONG_ADMIN/services/$service/plugins" | python -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'retry']
print('exists' if plugins else '')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'retry']
print('exists' if plugins else '')
" 2>/dev/null)
        
        if [ "$EXISTING" = "exists" ]; then
            echo "   ⚠️  Retry plugin already exists for $service"
        else
            echo "   ❌ Failed to create retry plugin for $service"
            echo "      Response: $(echo "$RESPONSE" | head -3)"
        fi
    fi
done

echo "✅ Retry logic configured"

# 3. Configure Health Checks (using PATCH)
echo ""
echo "3️⃣  Configuring Health Checks..."

# Ollama service health check
echo "   Configuring health check for ollama-service..."
OLLAMA_RESPONSE=$(curl -s -X PATCH "$KONG_ADMIN/services/ollama-service" \
  -H "Content-Type: application/json" \
  -d '{
    "healthchecks": {
      "active": {
        "type": "http",
        "http_path": "/api/tags",
        "timeout": 5,
        "concurrency": 10,
        "healthy": {
          "interval": 10,
          "successes": 3
        },
        "unhealthy": {
          "interval": 10,
          "http_failures": 3,
          "timeouts": 3
        }
      },
      "passive": {
        "type": "http",
        "healthy": {
          "http_statuses": [200, 201, 202, 204, 301, 302, 307, 308]
        },
        "unhealthy": {
          "http_statuses": [429, 500, 502, 503, 504],
          "timeouts": 5
        }
      }
    }
  }')

if echo "$OLLAMA_RESPONSE" | grep -q "healthchecks"; then
    echo "   ✅ Health check configured for ollama-service"
else
    echo "   ⚠️  Health check may already be configured or failed"
fi

# FastAPI service health check
echo "   Configuring health check for fastapi-service..."
FASTAPI_RESPONSE=$(curl -s -X PATCH "$KONG_ADMIN/services/fastapi-service" \
  -H "Content-Type: application/json" \
  -d '{
    "healthchecks": {
      "active": {
        "type": "http",
        "http_path": "/health",
        "timeout": 5,
        "concurrency": 10,
        "healthy": {
          "interval": 10,
          "successes": 3
        },
        "unhealthy": {
          "interval": 10,
          "http_failures": 3,
          "timeouts": 3
        }
      },
      "passive": {
        "type": "http",
        "healthy": {
          "http_statuses": [200, 201, 202, 204]
        },
        "unhealthy": {
          "http_statuses": [429, 500, 502, 503, 504],
          "timeouts": 5
        }
      }
    }
  }')

if echo "$FASTAPI_RESPONSE" | grep -q "healthchecks"; then
    echo "   ✅ Health check configured for fastapi-service"
else
    echo "   ⚠️  Health check may already be configured or failed"
fi

echo "✅ Health checks configured"

echo ""
echo "=============================================================="
echo "✅ Phase 4 manual setup complete!"
echo ""
echo "📊 Verify Configuration:"
echo "   curl http://localhost:8301/services/anthropic-service/plugins | grep proxy-cache"
echo "   curl http://localhost:8301/services/anthropic-service/plugins | grep retry"

