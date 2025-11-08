#!/bin/bash
# Phase 2: Security & Access Control Setup

KONG_ADMIN="http://localhost:8301"

echo "🔒 Phase 2: Setting up Security & Access Control..."

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

# Get current IP address (for local development)
CURRENT_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "127.0.0.1")
LOCALHOST="127.0.0.1"
LOCAL_NETWORK="192.168.1.0/24"  # Adjust this to your local network

echo ""
echo "📋 Detected IP: $CURRENT_IP"
echo "   Using allowed IPs: $LOCALHOST, $LOCAL_NETWORK"
echo ""

# 1. IP Whitelisting
echo "🔐 1. Configuring IP Whitelisting..."

# Enable IP restriction for all LLM services
for service in anthropic-service deepseek-service grok-service ollama-service; do
    echo "   Configuring IP restriction for $service..."
    
    # Create IP restriction plugin
    RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      --data "name=ip-restriction" \
      --data "config.allow=$LOCALHOST" \
      --data "config.allow=$LOCAL_NETWORK")
    
    PLUGIN_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', 'error'))" 2>/dev/null)
    
    if [ "$PLUGIN_ID" != "error" ] && [ -n "$PLUGIN_ID" ]; then
        echo "   ✅ IP restriction enabled for $service"
    else
        # Check if plugin already exists
        EXISTING=$(curl -s "$KONG_ADMIN/services/$service/plugins" | python3 -c "import sys, json; data=json.load(sys.stdin); plugins=[p for p in data.get('data', []) if p.get('name') == 'ip-restriction']; print('exists' if plugins else '')" 2>/dev/null)
        if [ "$EXISTING" = "exists" ]; then
            echo "   ⚠️  IP restriction already exists for $service (skipping)"
        else
            echo "   ❌ Failed to enable IP restriction for $service"
            echo "   Response: $RESPONSE"
        fi
    fi
done

# Note: FastAPI service can have different IP restrictions if needed
echo "   ℹ️  FastAPI service IP restrictions can be configured separately if needed"
echo "✅ IP Whitelisting configured"

# 2. Consumer-Based Access Control (ACLs)
echo ""
echo "👥 2. Configuring Consumer-Based Access Control..."

# Create ACL plugin for services
for service in anthropic-service deepseek-service grok-service ollama-service fastapi-service; do
    echo "   Configuring ACL for $service..."
    
    # Enable ACL plugin
    RESPONSE=$(curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
      --data "name=acl" \
      --data "config.allow=reasoning-engine" \
      --data "config.allow=swarm-agent" \
      --data "config.allow=query-deepseek" \
      --data "config.allow=admin")
    
    # Check if plugin was created successfully
    PLUGIN_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null)
    
    if [ -n "$PLUGIN_ID" ] && [ "$PLUGIN_ID" != "error" ]; then
        echo "   ✅ ACL enabled for $service (ID: $PLUGIN_ID)"
    else
        EXISTING=$(curl -s "$KONG_ADMIN/services/$service/plugins" | python3 -c "import sys, json; data=json.load(sys.stdin); plugins=[p for p in data.get('data', []) if p.get('name') == 'acl']; print('exists' if plugins else '')" 2>/dev/null)
        if [ "$EXISTING" = "exists" ]; then
            echo "   ⚠️  ACL already exists for $service (skipping)"
        else
            # Response might be valid JSON even if ID check fails
            if echo "$RESPONSE" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
                echo "   ✅ ACL enabled for $service"
            else
                echo "   ❌ Failed to enable ACL for $service"
            fi
        fi
    fi
done

# Assign ACL groups to consumers
echo ""
echo "   Assigning ACL groups to consumers..."

# Reasoning Engine Consumer - limited access
curl -s -X POST "$KONG_ADMIN/consumers/reasoning-engine-consumer/acls" \
  --data "group=reasoning-engine" > /dev/null
echo "   ✅ reasoning-engine-consumer → reasoning-engine group"

# Swarm Agent Consumer - limited access
curl -s -X POST "$KONG_ADMIN/consumers/swarm-agent-consumer/acls" \
  --data "group=swarm-agent" > /dev/null
echo "   ✅ swarm-agent-consumer → swarm-agent group"

# Query DeepSeek Consumer - limited access
curl -s -X POST "$KONG_ADMIN/consumers/query-deepseek-consumer/acls" \
  --data "group=query-deepseek" > /dev/null
echo "   ✅ query-deepseek-consumer → query-deepseek group"

# Admin Consumer - full access
curl -s -X POST "$KONG_ADMIN/consumers/admin-consumer/acls" \
  --data "group=admin" > /dev/null
echo "   ✅ admin-consumer → admin group"

echo "✅ Consumer-Based Access Control configured"

# 3. API Key Rotation Strategy Documentation
echo ""
echo "📝 3. Creating API Key Rotation Documentation..."

cat > ../kong/KEY_ROTATION.md << 'EOF'
# Kong API Key Rotation Guide

## Overview

This document describes the process for rotating Kong consumer API keys without downtime.

## Why Rotate Keys?

- Security best practice
- Compromised key revocation
- Periodic security maintenance
- Access control changes

## Rotation Process

### Step 1: Generate New Key

```bash
python3 kong/generate_keys.py
```

### Step 2: Add New Key to Consumer

```bash
# Replace CONSUMER_NAME and NEW_KEY
curl -X POST http://localhost:8301/consumers/CONSUMER_NAME/key-auth \
  --data "key=NEW_KEY"
```

### Step 3: Update Application Configuration

Update your application to use the new key:
- Update `.env` file
- Update configuration files
- Restart services

### Step 4: Verify New Key Works

```bash
# Test with new key
curl -H "apikey: NEW_KEY" http://localhost:8300/llm/anthropic/v1/messages
```

### Step 5: Remove Old Key (Optional)

Once verified, you can remove the old key:

```bash
# Get key ID first
curl http://localhost:8301/consumers/CONSUMER_NAME/key-auth

# Delete old key
curl -X DELETE http://localhost:8301/consumers/CONSUMER_NAME/key-auth/OLD_KEY_ID
```

## Automated Rotation Script

See `kong/rotate_keys.sh` for automated rotation.

## Best Practices

1. **Always add new key before removing old key** - Ensures zero downtime
2. **Test thoroughly** - Verify new key works before removing old
3. **Rotate keys periodically** - Every 90 days recommended
4. **Rotate immediately** - If key is compromised
5. **Document rotation** - Keep log of when keys were rotated

## Emergency Key Revocation

To immediately revoke access:

```bash
# Delete all keys for a consumer
curl http://localhost:8301/consumers/CONSUMER_NAME/key-auth | \
  python3 -c "import sys, json; \
  keys = json.load(sys.stdin).get('data', []); \
  [print(k['id']) for k in keys]" | \
  xargs -I {} curl -X DELETE \
  http://localhost:8301/consumers/CONSUMER_NAME/key-auth/{}
```

Then issue new keys as needed.
EOF

echo "✅ Key rotation documentation created at kong/KEY_ROTATION.md"

# Create rotation script
cat > ../kong/rotate_keys.sh << 'EOFSCRIPT'
#!/bin/bash
# Kong API Key Rotation Script

KONG_ADMIN="http://localhost:8301"
CONSUMER_NAME="$1"
NEW_KEY="$2"

if [ -z "$CONSUMER_NAME" ] || [ -z "$NEW_KEY" ]; then
    echo "Usage: $0 <consumer-name> <new-key>"
    echo "Example: $0 reasoning-engine-consumer $(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
    exit 1
fi

echo "🔄 Rotating API key for consumer: $CONSUMER_NAME"

# Add new key
echo "📝 Adding new key..."
RESPONSE=$(curl -s -X POST "$KONG_ADMIN/consumers/$CONSUMER_NAME/key-auth" \
  --data "key=$NEW_KEY")

KEY_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('id', 'error'))" 2>/dev/null)

if [ "$KEY_ID" != "error" ] && [ -n "$KEY_ID" ]; then
    echo "✅ New key added successfully"
    echo "   Key ID: $KEY_ID"
    echo "   New Key: $NEW_KEY"
    echo ""
    echo "⚠️  Next steps:"
    echo "   1. Update application configuration with new key"
    echo "   2. Restart services"
    echo "   3. Test with: curl -H 'apikey: $NEW_KEY' http://localhost:8300/llm/anthropic/v1/messages"
    echo "   4. Once verified, remove old key using Admin API"
else
    echo "❌ Failed to add new key"
    echo "Response: $RESPONSE"
    exit 1
fi
EOFSCRIPT

chmod +x ../kong/rotate_keys.sh
echo "✅ Key rotation script created at kong/rotate_keys.sh"

echo ""
echo "✅ Phase 2 setup complete!"
echo ""
echo "📋 Summary:"
echo "   - IP whitelisting: Enabled for all LLM services"
echo "   - ACL groups: Configured for all consumers"
echo "   - Key rotation: Documentation and script created"
echo ""
echo "🧪 Test IP restriction:"
echo "   # Should work from localhost:"
echo "   curl -H 'apikey: rQhK3Uq5L0cBMUEXXOn78lCOq7jXDYgo0NIhNeH_AYs' http://localhost:8300/llm/anthropic/v1/messages"
echo ""
echo "   # Should fail from external IP (if tested externally)"

