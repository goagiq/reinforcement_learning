# Ollama Network Security: Blocking Outbound Calls to China

## Problem Statement

Your locally hosted Ollama with DeepSeek model may be making outbound calls to:
- Alibaba Cloud (China)
- Other Chinese services
- Telemetry/analytics endpoints

**Question:** Can Kong Gateway block these outbound calls?

## Short Answer

**❌ Kong Gateway alone cannot directly block Ollama's outbound calls** because:
1. Kong is an API Gateway designed for **inbound traffic control** (requests coming TO your services)
2. Ollama runs independently and makes outbound calls directly (not through Kong)
3. Kong only sees traffic that routes through it (requests TO Ollama, not FROM Ollama)

## Understanding the Architecture

```
Your Application → Kong Gateway (port 8300) → Ollama (localhost:11434)
                                              ↓
                                        [Outbound calls to China]
                                        (Kong doesn't see these!)
```

**Current Flow:**
- Your app calls `http://localhost:8300/llm/ollama/...` (through Kong)
- Kong proxies to `http://host.docker.internal:11434` (Ollama)
- **But:** Ollama itself may make direct outbound calls that bypass Kong

## Solutions (Ranked by Effectiveness)

### Solution 1: Windows Firewall (RECOMMENDED) ✅

**Most effective** - Block outbound traffic at the OS level.

#### Quick Setup (PowerShell as Administrator)

```powershell
# Find Ollama process path
Get-Process ollama | Select-Object Path

# Block all outbound HTTPS/HTTP from Ollama
$ollamaPath = "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
# Or find the actual path using the command above

New-NetFirewallRule -DisplayName "Block Ollama Outbound Internet" `
    -Direction Outbound `
    -Program $ollamaPath `
    -Protocol TCP `
    -RemotePort 80,443 `
    -Action Block `
    -Enabled True

# Allow only localhost connections
New-NetFirewallRule -DisplayName "Allow Ollama Localhost" `
    -Direction Outbound `
    -Program $ollamaPath `
    -RemoteAddress 127.0.0.1,localhost `
    -Action Allow `
    -Enabled True
```

#### Block Specific China IP Ranges

```powershell
# China IP ranges (common ones)
$chinaRanges = @(
    "47.0.0.0/8",      # Alibaba Cloud
    "120.0.0.0/8",     # China Telecom
    "180.0.0.0/8",     # China Unicom
    "223.0.0.0/8"      # China Mobile
)

foreach ($range in $chinaRanges) {
    New-NetFirewallRule -DisplayName "Block China $range" `
        -Direction Outbound `
        -RemoteAddress $range `
        -Action Block `
        -Enabled True
}
```

### Solution 2: Docker Network Isolation (If Ollama is in Docker) ✅

Isolate Ollama in a Docker network with no internet access.

```yaml
# docker-compose.yml addition
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    networks:
      - ollama-internal-only  # No external access
    # Keep port 11434 for local access
    
networks:
  ollama-internal-only:
    driver: bridge
    internal: true  # This blocks external internet access!
```

**Note:** Your Kong setup already routes to `host.docker.internal:11434`, so this works if Ollama is running on the host (not in Docker).

### Solution 3: Monitor First, Then Block 🔍

**Before blocking, verify what Ollama is actually calling:**

#### Windows: Monitor Network Activity

```powershell
# Check active connections
netstat -ano | findstr ollama

# Or use Resource Monitor (Windows built-in)
# Task Manager → Performance → Open Resource Monitor → Network tab

# Monitor with Process Monitor (Sysinternals)
# Download from: https://docs.microsoft.com/en-us/sysinternals/downloads/procmon
```

#### Check Ollama Logs

```bash
# If Ollama is in Docker
docker logs ollama-container

# Or check Windows Event Viewer
# Windows Logs → Application → Look for Ollama entries
```

### Solution 4: Kong Proxy with Request Inspection (Limited Effectiveness) ⚠️

Kong can inspect/modify requests **TO** Ollama, but not **FROM** Ollama.

**However**, you can:
1. Route ALL Ollama traffic through Kong (already done ✅)
2. Use Kong's `request-transformer` plugin to strip headers that might trigger external calls
3. Monitor requests going through Kong

```yaml
# kong.yml - Add to ollama-service plugins
plugins:
  - name: request-transformer
    config:
      remove:
        headers:
          - "X-Forwarded-For"
          - "X-Real-IP"
          - "X-Telemetry"  # If any telemetry headers exist
```

**Limitation:** This only works for traffic going TO Ollama, not FROM Ollama.

### Solution 5: Offline Mode Configuration 📴

Check if Ollama/DeepSeek has environment variables for offline mode:

```bash
# Set environment variables before starting Ollama
set OLLAMA_NO_TELEMETRY=1
set OLLAMA_HOST=127.0.0.1:11434

# Or in PowerShell
$env:OLLAMA_NO_TELEMETRY=1
$env:OLLAMA_HOST="127.0.0.1:11434"
```

## Recommended Approach for Your Setup

Since you're running:
- **Kong:** In Docker (ports 8300/8301)
- **Ollama:** Locally on Windows (port 11434)
- **DeepSeek:** Model loaded in Ollama

### Phase 1: Monitor (Do This First) 🔍

1. **Check if DeepSeek actually makes outbound calls:**
   ```powershell
   # Monitor network activity while using DeepSeek
   netstat -ano | findstr ollama
   
   # Or use Resource Monitor
   # Windows Key → Type "Resource Monitor" → Network tab → Filter by "ollama"
   ```

2. **Check Ollama logs:**
   ```powershell
   # If Ollama runs as a service, check Windows Event Viewer
   # Or check logs in: %USERPROFILE%\.ollama\logs\
   ```

3. **Test with a simple query:**
   ```bash
   # Make a request through Kong
   curl http://localhost:8300/llm/ollama/api/chat \
     -H "apikey: YOUR_KEY" \
     -d '{"model":"deepseek-r1:8b","messages":[{"role":"user","content":"test"}]}'
   
   # While this runs, monitor network connections
   ```

### Phase 2: Block (If Needed) 🛡️

If you confirm outbound calls are happening:

#### Option A: Windows Firewall (Recommended)

```powershell
# Run PowerShell as Administrator

# 1. Find Ollama executable path
$ollama = Get-Process ollama -ErrorAction SilentlyContinue
if ($ollama) {
    $ollamaPath = $ollama.Path
    Write-Host "Found Ollama at: $ollamaPath"
} else {
    # Common locations
    $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaPath)) {
        Write-Host "Please specify Ollama path manually"
        exit
    }
}

# 2. Block outbound HTTPS/HTTP
New-NetFirewallRule -DisplayName "Block Ollama Outbound Internet" `
    -Direction Outbound `
    -Program $ollamaPath `
    -Protocol TCP `
    -RemotePort 80,443 `
    -Action Block `
    -Enabled True `
    -ErrorAction SilentlyContinue

# 3. Allow localhost (for Kong to access Ollama)
New-NetFirewallRule -DisplayName "Allow Ollama Localhost" `
    -Direction Outbound `
    -Program $ollamaPath `
    -RemoteAddress 127.0.0.1,localhost,::1 `
    -Action Allow `
    -Enabled True `
    -ErrorAction SilentlyContinue

Write-Host "✅ Firewall rules created!"
Write-Host "   - Blocked: Outbound HTTP/HTTPS"
Write-Host "   - Allowed: Localhost connections"
```

#### Option B: Block Specific China IP Ranges

```powershell
# Block known China/Alibaba IP ranges
$chinaRanges = @(
    "47.0.0.0/8",      # Alibaba Cloud
    "120.0.0.0/8",     # China Telecom
    "180.0.0.0/8",     # China Unicom  
    "223.0.0.0/8"      # China Mobile
)

foreach ($range in $chinaRanges) {
    New-NetFirewallRule -DisplayName "Block China $range" `
        -Direction Outbound `
        -RemoteAddress $range `
        -Action Block `
        -Enabled True `
        -ErrorAction SilentlyContinue
}

Write-Host "✅ Blocked China IP ranges"
```

### Phase 3: Verify ✅

After blocking, verify:

1. **Test DeepSeek still works:**
   ```bash
   curl http://localhost:8300/llm/ollama/api/chat \
     -H "apikey: YOUR_KEY" \
     -d '{"model":"deepseek-r1:8b","messages":[{"role":"user","content":"Hello"}]}'
   ```

2. **Check for blocked attempts:**
   ```powershell
   # View firewall rules
   Get-NetFirewallRule -DisplayName "*Ollama*" | Format-Table DisplayName, Enabled, Direction, Action
   
   # Check Windows Firewall logs (if logging enabled)
   # Windows Defender Firewall → Advanced Settings → Properties → Logging
   ```

3. **Monitor network activity:**
   ```powershell
   # Should show no external connections
   netstat -ano | findstr ollama
   ```

## Additional Considerations

1. **Model Updates:** If you block internet access, Ollama won't be able to download model updates. This is fine for local-only operation.

2. **Performance:** Local-only operation should be faster (no external latency).

3. **Security:** Blocking outbound calls prevents data exfiltration and unauthorized communications.

4. **Compliance:** Ensure blocking aligns with your organization's policies.

## Kong Gateway Limitations

**Kong cannot block Ollama's outbound calls because:**

- Kong is a **reverse proxy** (handles inbound requests)
- Ollama makes **direct outbound calls** (bypasses Kong)
- Kong only sees traffic that **routes through it**

**However, Kong provides:**
- ✅ Inbound traffic security (API keys, rate limiting)
- ✅ Request/response logging
- ✅ Traffic monitoring (Prometheus metrics)
- ❌ Cannot control outbound traffic from backend services

## Summary

| Solution | Effectiveness | Complexity | Recommended |
|----------|---------------|------------|-------------|
| Windows Firewall | ⭐⭐⭐⭐⭐ | Low | ✅ **Yes** |
| Monitor First | ⭐⭐⭐⭐ | Low | ✅ **Yes** (do this first!) |
| Docker Network Isolation | ⭐⭐⭐⭐⭐ | Medium | ✅ Yes (if using Docker) |
| Kong Proxy | ⭐⭐ | Low | ⚠️ Limited |
| Offline Mode | ⭐⭐⭐ | Low | ✅ Maybe |

## Next Steps

1. **Monitor Ollama's network activity** to confirm outbound calls
2. **Implement Windows Firewall rules** to block outbound traffic
3. **Test** that DeepSeek still works locally
4. **Document** your firewall rules for future reference

## Quick Reference Script

Save this as `block-ollama-outbound.ps1` and run as Administrator:

```powershell
# block-ollama-outbound.ps1
# Run as Administrator

$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

if (-not (Test-Path $ollamaPath)) {
    Write-Host "❌ Ollama not found at $ollamaPath"
    Write-Host "Please update the path in this script"
    exit 1
}

# Block outbound HTTP/HTTPS
New-NetFirewallRule -DisplayName "Block Ollama Outbound Internet" `
    -Direction Outbound `
    -Program $ollamaPath `
    -Protocol TCP `
    -RemotePort 80,443 `
    -Action Block `
    -Enabled True `
    -ErrorAction SilentlyContinue

# Allow localhost
New-NetFirewallRule -DisplayName "Allow Ollama Localhost" `
    -Direction Outbound `
    -Program $ollamaPath `
    -RemoteAddress 127.0.0.1,localhost,::1 `
    -Action Allow `
    -Enabled True `
    -ErrorAction SilentlyContinue

Write-Host "✅ Firewall rules created!"
Write-Host "   Ollama can only connect to localhost"
Write-Host "   All external internet access blocked"
```

---

**Note:** Kong Gateway is excellent for protecting inbound traffic, but for outbound traffic control, network-level solutions (firewall, Docker network isolation) are more effective.
