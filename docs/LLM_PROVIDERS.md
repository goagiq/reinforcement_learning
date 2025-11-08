# LLM Provider Configuration Guide

The reasoning engine now supports multiple LLM providers for flexibility and performance.

## Supported Providers

1. **Ollama** (Local) - Default, free, runs locally
2. **DeepSeek Cloud** - Fast cloud API, requires subscription
3. **Grok** (xAI) - Fast cloud API, requires subscription

## Configuration

Edit `configs/train_config_full.yaml`:

```yaml
reasoning:
  enabled: true
  provider: "ollama"                    # Options: "ollama", "deepseek_cloud", "grok"
  model: "deepseek-r1:8b"              # Model name (provider-specific)
  api_key: null                        # API key (set via environment variable for security)
  base_url: null                       # Optional, uses provider defaults
  pre_trade_validation: true
  confidence_threshold: 0.7
  timeout: 2.0                         # Timeout in seconds
```

## Provider-Specific Setup

### 1. Ollama (Local - Default)

**Pros:** Free, private, no API costs  
**Cons:** Requires local GPU, slower than cloud

**Setup:**
```bash
# Install Ollama (if not already installed)
# Download from: https://ollama.ai

# Pull DeepSeek-R1 model
ollama pull deepseek-r1:8b

# Verify it's running
ollama list
```

**Config:**
```yaml
reasoning:
  provider: "ollama"
  model: "deepseek-r1:8b"
  base_url: "http://localhost:11434"  # Optional, this is default
```

### 2. DeepSeek Cloud API

**Pros:** Fast, no local GPU needed, reliable  
**Cons:** Requires subscription, API costs

**Setup:**
1. Sign up at: https://platform.deepseek.com
2. Get your API key from the dashboard
3. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:DEEPSEEK_API_KEY="your-api-key-here"
   
   # Windows (CMD)
   set DEEPSEEK_API_KEY=your-api-key-here
   
   # Linux/Mac
   export DEEPSEEK_API_KEY=your-api-key-here
   ```

**Config:**
```yaml
reasoning:
  provider: "deepseek_cloud"
  model: "deepseek-chat"                # DeepSeek Cloud model name
  api_key: null                         # Will use DEEPSEEK_API_KEY env var
  base_url: null                        # Uses https://api.deepseek.com by default
```

**Available Models:**
- `deepseek-chat` - Standard chat model
- `deepseek-coder` - Code-focused model

### 3. Grok (xAI)

**Pros:** Fast, good reasoning, real-time data access  
**Cons:** Requires X/Twitter account, API costs

**Setup:**
1. Sign up for Grok API: https://x.ai/api
2. Get your API key
3. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:GROK_API_KEY="your-api-key-here"
   
   # Windows (CMD)
   set GROK_API_KEY=your-api-key-here
   
   # Linux/Mac
   export GROK_API_KEY=your-api-key-here
   ```

**Config:**
```yaml
reasoning:
  provider: "grok"
  model: "grok-beta"                    # Grok model name
  api_key: null                         # Will use GROK_API_KEY env var
  base_url: null                        # Uses https://api.x.ai by default
```

**Available Models:**
- `grok-beta` - Standard Grok model
- `grok-2` - Latest Grok model (if available)

## Environment Variables

For security, **never commit API keys to git**. Use environment variables:

```bash
# DeepSeek Cloud
export DEEPSEEK_API_KEY="sk-..."

# Grok
export GROK_API_KEY="xai-..."
```

The system will automatically check:
1. Config file (`api_key` field)
2. `DEEPSEEK_API_KEY` environment variable (for DeepSeek)
3. `GROK_API_KEY` environment variable (for Grok)

## Switching Providers

Simply change the `provider` field in config:

```yaml
# Switch to DeepSeek Cloud
reasoning:
  provider: "deepseek_cloud"
  model: "deepseek-chat"

# Switch to Grok
reasoning:
  provider: "grok"
  model: "grok-beta"

# Switch back to Ollama
reasoning:
  provider: "ollama"
  model: "deepseek-r1:8b"
```

## Performance Comparison

| Provider | Speed | Cost | Privacy | Setup Complexity |
|----------|-------|------|---------|------------------|
| Ollama | Slow-Medium | Free | High | Medium |
| DeepSeek Cloud | Fast | Paid | Low | Easy |
| Grok | Fast | Paid | Low | Easy |

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
ollama list

# Restart Ollama if needed
# Windows: Restart Ollama service
# Linux/Mac: ollama serve
```

### API Key Not Found
- Ensure environment variable is set correctly
- Check variable name matches (`DEEPSEEK_API_KEY` or `GROK_API_KEY`)
- Restart the application after setting environment variable

### Timeout Errors
- Increase `timeout` in config (default: 2.0 seconds)
- Check network connectivity for cloud providers
- For Ollama, ensure local GPU has enough memory

### Model Not Found
- **Ollama**: Run `ollama pull <model-name>`
- **DeepSeek Cloud**: Use correct model name (`deepseek-chat`, not `deepseek-r1:8b`)
- **Grok**: Use `grok-beta` or check available models in API docs

## Example Usage

```python
from src.reasoning_engine import ReasoningEngine

# Using Ollama (default)
engine = ReasoningEngine(
    provider_type="ollama",
    model="deepseek-r1:8b"
)

# Using DeepSeek Cloud
engine = ReasoningEngine(
    provider_type="deepseek_cloud",
    model="deepseek-chat",
    api_key="sk-..."  # Or use DEEPSEEK_API_KEY env var
)

# Using Grok
engine = ReasoningEngine(
    provider_type="grok",
    model="grok-beta",
    api_key="xai-..."  # Or use GROK_API_KEY env var
)
```

## Security Best Practices

1. **Never commit API keys** to git
2. Use environment variables for API keys
3. Rotate API keys regularly
4. Use different keys for development/production
5. Monitor API usage to detect anomalies

## Cost Considerations

- **Ollama**: Free (uses your own hardware)
- **DeepSeek Cloud**: Pay-per-use, check pricing at https://platform.deepseek.com
- **Grok**: Pay-per-use, check pricing at https://x.ai/api

For high-frequency trading, cloud APIs can get expensive. Consider:
- Using Ollama for development/testing
- Using cloud APIs for production
- Implementing request caching to reduce API calls
- Using batch processing when possible

