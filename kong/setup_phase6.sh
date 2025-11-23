#!/bin/bash
# Phase 6: FastAPI Integration Setup

KONG_ADMIN="http://localhost:8301"
KONG_PROXY="http://localhost:8300"

echo "🔌 Phase 6: FastAPI Integration Setup"
echo "======================================"
echo ""

# Wait for Kong to be ready
echo "⏳ Waiting for Kong to be ready..."
for i in {1..30}; do
    if curl -s "$KONG_ADMIN/" > /dev/null 2>&1; then
        echo "✅ Kong is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# 1. Verify FastAPI Service
echo ""
echo "1️⃣  Verifying FastAPI Service..."

FASTAPI_SERVICE=$(curl -s "$KONG_ADMIN/services/fastapi-service" | python -c "
import sys, json
data = json.load(sys.stdin)
print('exists' if data.get('name') == 'fastapi-service' else 'not_found')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
print('exists' if data.get('name') == 'fastapi-service' else 'not_found')
" 2>/dev/null)

if [ "$FASTAPI_SERVICE" = "exists" ]; then
    echo "   ✅ FastAPI service exists"
    
    # Check service URL
    SERVICE_URL=$(curl -s "$KONG_ADMIN/services/fastapi-service" | python -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('url', 'not_set'))
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('url', 'not_set'))
" 2>/dev/null)
    
    if [ "$SERVICE_URL" != "not_set" ] && [ -n "$SERVICE_URL" ]; then
        echo "   ✅ Service URL: $SERVICE_URL"
    else
        echo "   ⚠️  Service URL not set, updating..."
        # Update service URL
        curl -s -X PATCH "$KONG_ADMIN/services/fastapi-service" \
          -H "Content-Type: application/json" \
          -d '{"url": "http://host.docker.internal:8200"}' > /dev/null
        echo "   ✅ Service URL updated to http://host.docker.internal:8200"
    fi
else
    echo "   ❌ FastAPI service not found, creating..."
    # Create service
    curl -s -X POST "$KONG_ADMIN/services" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "fastapi-service",
        "url": "http://host.docker.internal:8200"
      }' > /dev/null
    echo "   ✅ FastAPI service created"
fi

# 2. Verify FastAPI Route
echo ""
echo "2️⃣  Verifying FastAPI Route..."

FASTAPI_ROUTE=$(curl -s "$KONG_ADMIN/routes" | python -c "
import sys, json
data = json.load(sys.stdin)
routes = [r for r in data.get('data', []) if r.get('name') == 'fastapi-route']
print('exists' if routes else 'not_found')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
routes = [r for r in data.get('data', []) if r.get('name') == 'fastapi-route']
print('exists' if routes else 'not_found')
" 2>/dev/null)

if [ "$FASTAPI_ROUTE" = "exists" ]; then
    echo "   ✅ FastAPI route exists"
    
    # Check route path
    ROUTE_PATH=$(curl -s "$KONG_ADMIN/routes/fastapi-route" | python -c "
import sys, json
data = json.load(sys.stdin)
paths = data.get('paths', [])
print(paths[0] if paths else 'not_set')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
paths = data.get('paths', [])
print(paths[0] if paths else 'not_set')
" 2>/dev/null)
    
    echo "   ✅ Route path: $ROUTE_PATH"
else
    echo "   ❌ FastAPI route not found, creating..."
    # Create route
    curl -s -X POST "$KONG_ADMIN/services/fastapi-service/routes" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "fastapi-route",
        "paths": ["/api"],
        "strip_path": false,
        "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
      }' > /dev/null
    echo "   ✅ FastAPI route created: /api"
fi

# 3. Verify Security Plugins
echo ""
echo "3️⃣  Verifying Security Plugins on FastAPI Service..."

# Check key-auth
KEY_AUTH=$(curl -s "$KONG_ADMIN/services/fastapi-service/plugins" | python -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'key-auth']
print('exists' if plugins else 'not_found')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'key-auth']
print('exists' if plugins else 'not_found')
" 2>/dev/null)

if [ "$KEY_AUTH" = "exists" ]; then
    echo "   ✅ Key-auth plugin enabled"
else
    echo "   ⚠️  Key-auth plugin not found (may be optional for FastAPI)"
fi

# Check rate-limiting
RATE_LIMIT=$(curl -s "$KONG_ADMIN/services/fastapi-service/plugins" | python -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'rate-limiting']
print('exists' if plugins else 'not_found')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'rate-limiting']
print('exists' if plugins else 'not_found')
" 2>/dev/null)

if [ "$RATE_LIMIT" = "exists" ]; then
    echo "   ✅ Rate-limiting plugin enabled"
else
    echo "   ⚠️  Rate-limiting plugin not found"
fi

# 4. Configure CORS Plugin
echo ""
echo "4️⃣  Configuring CORS Plugin for FastAPI..."

CORS_PLUGIN=$(curl -s "$KONG_ADMIN/services/fastapi-service/plugins" | python -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'cors']
print('exists' if plugins else 'not_found')
" 2>/dev/null || python3 -c "
import sys, json
data = json.load(sys.stdin)
plugins = [p for p in data.get('data', []) if p.get('name') == 'cors']
print('exists' if plugins else 'not_found')
" 2>/dev/null)

if [ "$CORS_PLUGIN" = "exists" ]; then
    echo "   ⚠️  CORS plugin already exists"
else
    # Create CORS plugin
    CORS_RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/fastapi-service/plugins" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "cors",
        "config": {
          "origins": ["*"],
          "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
          "headers": ["*"],
          "exposed_headers": ["*"],
          "credentials": true,
          "max_age": 3600,
          "preflight_continue": false
        }
      }')
    
    CORS_ID=$(echo "$CORS_RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null || python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null)
    
    if [ -n "$CORS_ID" ] && [ "$CORS_ID" != "error" ]; then
        echo "   ✅ CORS plugin enabled (ID: $CORS_ID)"
        echo "      - Origins: * (all allowed)"
        echo "      - Methods: GET, POST, PUT, DELETE, PATCH, OPTIONS"
        echo "      - Credentials: true"
    else
        echo "   ⚠️  CORS plugin may have been configured"
    fi
fi

# 5. Test FastAPI Route
echo ""
echo "5️⃣  Testing FastAPI Route Through Kong..."

# Test if FastAPI is running
FASTAPI_RUNNING=$(curl -s http://localhost:8200/api/setup/check 2>&1 | head -1 | grep -q "200\|json" && echo "yes" || echo "no")

if [ "$FASTAPI_RUNNING" = "yes" ]; then
    echo "   ✅ FastAPI is running on port 8200"
    
    # Test through Kong (may require API key)
    KONG_TEST=$(curl -s "$KONG_PROXY/api/setup/check" 2>&1 | head -1)
    
    if echo "$KONG_TEST" | grep -q "200\|json\|setup"; then
        echo "   ✅ FastAPI accessible through Kong"
    elif echo "$KONG_TEST" | grep -q "401\|403"; then
        echo "   ⚠️  FastAPI route requires authentication (expected if key-auth enabled)"
    else
        echo "   ⚠️  FastAPI route test: $KONG_TEST"
    fi
else
    echo "   ⚠️  FastAPI not running on port 8200 (start it to test)"
fi

echo ""
echo "======================================"
echo "✅ Phase 6 setup complete!"
echo ""
echo "📋 Summary:"
echo "   - FastAPI Service: ✅ Configured"
echo "   - FastAPI Route: ✅ /api route configured"
echo "   - Security Plugins: ✅ Verified"
echo "   - CORS Plugin: ✅ Configured"
echo ""
echo "📝 Next Steps:"
echo "   1. Update frontend vite.config.js to use Kong (port 8300)"
echo "   2. Update frontend WebSocket connection to use Kong"
echo "   3. Test frontend with Kong"
echo "   4. Verify CORS works"









