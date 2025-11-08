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
