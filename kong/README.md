# Kong Gateway Setup

This directory contains the Kong Gateway configuration for the NT8 RL Trading System.

## Prerequisites

- Docker and Docker Compose installed
- Ports 8300, 8301, and 5434 available (Kong PostgreSQL uses 5434 to avoid conflicts)

## Quick Start

### 1. Generate API Keys

First, generate secure API keys for each consumer:

```bash
# Generate keys (you can use any secure method)
python3 -c "import secrets; print('REASONING_ENGINE_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('SWARM_AGENT_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('QUERY_DEEPSEEK_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('ADMIN_KEY=' + secrets.token_urlsafe(32))"
```

Update `kong.yml` with the generated keys, replacing the `PLACEHOLDER` values.

### 2. Start Kong

```bash
cd kong
docker-compose up -d
```

### 3. Verify Kong is Running

```bash
# Check Kong status
curl http://localhost:8301/

# Check services
curl http://localhost:8301/services

# Check consumers
curl http://localhost:8301/consumers
```

### 4. Test a Route

```bash
# Test Anthropic route (requires API key)
curl -H "apikey: YOUR_KEY_HERE" http://localhost:8300/llm/anthropic/v1/messages
```

## Configuration

### Services

- **anthropic-service**: Routes to Anthropic API
- **deepseek-service**: Routes to DeepSeek Cloud API
- **grok-service**: Routes to Grok (xAI) API
- **ollama-service**: Routes to local Ollama instance
- **fastapi-service**: Routes to FastAPI server

### Consumers

- **reasoning-engine-consumer**: For ReasoningEngine
- **swarm-agent-consumer**: For Swarm Agents
- **query-deepseek-consumer**: For QueryDeepSeek
- **admin-consumer**: For admin operations

## Admin API

Kong Admin API is available at: `http://localhost:8301`

### Useful Commands

```bash
# List all services
curl http://localhost:8301/services

# List all routes
curl http://localhost:8301/routes

# List all consumers
curl http://localhost:8301/consumers

# Get consumer details
curl http://localhost:8301/consumers/reasoning-engine-consumer

# List plugins for a service
curl http://localhost:8301/services/anthropic-service/plugins
```

## Stopping Kong

```bash
docker-compose down
```

To remove volumes (database data):

```bash
docker-compose down -v
```

## Troubleshooting

### Check Kong logs

```bash
docker-compose logs kong
```

### Check PostgreSQL logs

```bash
docker-compose logs postgres
```

### Restart Kong

```bash
docker-compose restart kong
```

### Verify database connection

```bash
docker-compose exec kong kong health
```

