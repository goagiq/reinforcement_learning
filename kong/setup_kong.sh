#!/bin/bash
# Setup Kong Gateway - Create services, routes, and consumers

KONG_ADMIN="http://localhost:8301"
KONG_PROXY="http://localhost:8300"

echo "🚀 Setting up Kong Gateway..."

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

# Create Services
echo ""
echo "📦 Creating services..."

# Anthropic Service
curl -s -X POST "$KONG_ADMIN/services" \
  --data "name=anthropic-service" \
  --data "url=https://api.anthropic.com" | jq -r '.id // "exists or error"'

# DeepSeek Service
curl -s -X POST "$KONG_ADMIN/services" \
  --data "name=deepseek-service" \
  --data "url=https://api.deepseek.com" | jq -r '.id // "exists or error"'

# Grok Service
curl -s -X POST "$KONG_ADMIN/services" \
  --data "name=grok-service" \
  --data "url=https://api.x.ai" | jq -r '.id // "exists or error"'

# Ollama Service
curl -s -X POST "$KONG_ADMIN/services" \
  --data "name=ollama-service" \
  --data "url=http://host.docker.internal:11434" | jq -r '.id // "exists or error"'

# FastAPI Service
curl -s -X POST "$KONG_ADMIN/services" \
  --data "name=fastapi-service" \
  --data "url=http://host.docker.internal:8200" | jq -r '.id // "exists or error"'

echo "✅ Services created"

# Create Routes
echo ""
echo "🛣️  Creating routes..."

# Anthropic Route
curl -s -X POST "$KONG_ADMIN/services/anthropic-service/routes" \
  --data "name=anthropic-route" \
  --data "paths[]=/llm/anthropic" \
  --data "strip_path=true" | jq -r '.id // "exists or error"'

# DeepSeek Route
curl -s -X POST "$KONG_ADMIN/services/deepseek-service/routes" \
  --data "name=deepseek-route" \
  --data "paths[]=/llm/deepseek" \
  --data "strip_path=true" | jq -r '.id // "exists or error"'

# Grok Route
curl -s -X POST "$KONG_ADMIN/services/grok-service/routes" \
  --data "name=grok-route" \
  --data "paths[]=/llm/grok" \
  --data "strip_path=true" | jq -r '.id // "exists or error"'

# Ollama Route
curl -s -X POST "$KONG_ADMIN/services/ollama-service/routes" \
  --data "name=ollama-route" \
  --data "paths[]=/llm/ollama" \
  --data "strip_path=true" | jq -r '.id // "exists or error"'

# FastAPI Route
curl -s -X POST "$KONG_ADMIN/services/fastapi-service/routes" \
  --data "name=fastapi-route" \
  --data "paths[]=/api" \
  --data "strip_path=false" | jq -r '.id // "exists or error"'

echo "✅ Routes created"

# Create Consumers
echo ""
echo "👥 Creating consumers..."

# Reasoning Engine Consumer
REASONING_KEY="rQhK3Uq5L0cBMUEXXOn78lCOq7jXDYgo0NIhNeH_AYs"
curl -s -X POST "$KONG_ADMIN/consumers" \
  --data "username=reasoning-engine-consumer" | jq -r '.id // "exists or error"'
curl -s -X POST "$KONG_ADMIN/consumers/reasoning-engine-consumer/key-auth" \
  --data "key=$REASONING_KEY" | jq -r '.id // "exists or error"'

# Swarm Agent Consumer
SWARM_KEY="W-1--OrRPg-J6JmYZKM_lk5AjeNo-cICkFEL5ieihnw"
curl -s -X POST "$KONG_ADMIN/consumers" \
  --data "username=swarm-agent-consumer" | jq -r '.id // "exists or error"'
curl -s -X POST "$KONG_ADMIN/consumers/swarm-agent-consumer/key-auth" \
  --data "key=$SWARM_KEY" | jq -r '.id // "exists or error"'

# Query DeepSeek Consumer
DEEPSEEK_KEY="guqhYjH70oDGQn6uiBPCn1tpt4ZGP8Qlmh3CyU933Rs"
curl -s -X POST "$KONG_ADMIN/consumers" \
  --data "username=query-deepseek-consumer" | jq -r '.id // "exists or error"'
curl -s -X POST "$KONG_ADMIN/consumers/query-deepseek-consumer/key-auth" \
  --data "key=$DEEPSEEK_KEY" | jq -r '.id // "exists or error"'

# Admin Consumer
ADMIN_KEY="EhJ2T5SpLeqUAaFxkBwoWcnlg1T_5AappZ9VOhXzgXI"
curl -s -X POST "$KONG_ADMIN/consumers" \
  --data "username=admin-consumer" | jq -r '.id // "exists or error"'
curl -s -X POST "$KONG_ADMIN/consumers/admin-consumer/key-auth" \
  --data "key=$ADMIN_KEY" | jq -r '.id // "exists or error"'

echo "✅ Consumers created"

# Enable Key Auth Plugin for all services
echo ""
echo "🔐 Enabling key-auth plugin..."

for service in anthropic-service deepseek-service grok-service ollama-service fastapi-service; do
    curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      --data "name=key-auth" | jq -r '.id // "exists or error"'
done

echo "✅ Key-auth plugin enabled"

# Enable Rate Limiting Plugin
echo ""
echo "⏱️  Enabling rate-limiting plugin..."

# Anthropic: 1000/min, 10000/hour, 100000/day
curl -s -X POST "$KONG_ADMIN/services/anthropic-service/plugins" \
  --data "name=rate-limiting" \
  --data "config.minute=1000" \
  --data "config.hour=10000" \
  --data "config.day=100000" \
  --data "config.policy=local" | jq -r '.id // "exists or error"'

# DeepSeek: 2000/min, 20000/hour, 200000/day
curl -s -X POST "$KONG_ADMIN/services/deepseek-service/plugins" \
  --data "name=rate-limiting" \
  --data "config.minute=2000" \
  --data "config.hour=20000" \
  --data "config.day=200000" \
  --data "config.policy=local" | jq -r '.id // "exists or error"'

# Grok: 1500/min, 15000/hour, 150000/day
curl -s -X POST "$KONG_ADMIN/services/grok-service/plugins" \
  --data "name=rate-limiting" \
  --data "config.minute=1500" \
  --data "config.hour=15000" \
  --data "config.day=150000" \
  --data "config.policy=local" | jq -r '.id // "exists or error"'

# Ollama: 5000/min, 50000/hour, 500000/day
curl -s -X POST "$KONG_ADMIN/services/ollama-service/plugins" \
  --data "name=rate-limiting" \
  --data "config.minute=5000" \
  --data "config.hour=50000" \
  --data "config.day=500000" \
  --data "config.policy=local" | jq -r '.id // "exists or error"'

# FastAPI: 10000/min, 100000/hour, 1000000/day
curl -s -X POST "$KONG_ADMIN/services/fastapi-service/plugins" \
  --data "name=rate-limiting" \
  --data "config.minute=10000" \
  --data "config.hour=100000" \
  --data "config.day=1000000" \
  --data "config.policy=local" | jq -r '.id // "exists or error"'

echo "✅ Rate-limiting plugin enabled"

echo ""
echo "✅ Kong Gateway setup complete!"
echo ""
echo "📋 Summary:"
echo "   - Services: 5 (anthropic, deepseek, grok, ollama, fastapi)"
echo "   - Routes: 5"
echo "   - Consumers: 4 (reasoning-engine, swarm-agent, query-deepseek, admin)"
echo "   - Plugins: key-auth, rate-limiting"
echo ""
echo "🔑 API Keys (save these securely!):"
echo "   Reasoning Engine: $REASONING_KEY"
echo "   Swarm Agent: $SWARM_KEY"
echo "   Query DeepSeek: $DEEPSEEK_KEY"
echo "   Admin: $ADMIN_KEY"
echo ""
echo "🧪 Test Kong:"
echo "   curl -H 'apikey: $REASONING_KEY' $KONG_PROXY/llm/anthropic/v1/messages"

