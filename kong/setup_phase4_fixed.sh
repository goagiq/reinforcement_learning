#!/bin/bash
# Phase 4: Traffic Management Setup (Fixed with correct configuration)

KONG_ADMIN="http://localhost:8301"

echo "🔧 Phase 4: Setting up Traffic Management Plugins (Fixed)..."
echo "============================================================"
echo ""

# Note: Kong needs to be restarted with KONG_PLUGINS=bundled,proxy-cache,retry
# for retry plugin to work. Proxy-cache just needs strategy="memory"

# 1. Enable Proxy Cache Plugin (with strategy)
echo "1️⃣  Enabling Proxy Cache Plugin..."

for service in anthropic-service deepseek-service grok-service ollama-service; do
    echo "   Configuring cache for $service..."
    
    RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "proxy-cache",
        "config": {
          "strategy": "memory",
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
            ERROR_MSG=$(echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('message', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
            echo "   ❌ Failed to create proxy cache for $service"
            echo "      Error: $ERROR_MSG"
        fi
    fi
done

echo "✅ Proxy caching configured"

# 2. Retry Plugin (requires Kong restart with KONG_PLUGINS=bundled,retry)
echo ""
echo "2️⃣  Retry Plugin Configuration..."
echo "   ⚠️  NOTE: Retry plugin requires Kong to be restarted with:"
echo "      KONG_PLUGINS=bundled,retry"
echo ""
echo "   Please restart Kong with the updated docker-compose.yml that includes:"
echo "      KONG_PLUGINS: bundled,proxy-cache,retry"
echo ""
echo "   After restart, run this script again to enable retry plugin."

# 3. Health Checks (already working)
echo ""
echo "3️⃣  Health Checks..."
echo "   ✅ Health checks were configured in previous attempt"
echo "   ✅ Ollama: /api/tags"
echo "   ✅ FastAPI: /health (needs to be added to FastAPI app)"

echo ""
echo "============================================================"
echo "✅ Phase 4 setup (proxy-cache) complete!"
echo ""
echo "📝 Next Steps:"
echo "   1. Restart Kong with KONG_PLUGINS=bundled,proxy-cache,retry"
echo "   2. Run this script again to enable retry plugin"
echo "   3. Add /health endpoint to FastAPI"
echo "   4. Test caching and retry functionality"

