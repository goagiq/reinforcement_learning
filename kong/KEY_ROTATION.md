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
