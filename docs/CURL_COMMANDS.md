# Curl Commands for Performance Monitoring

## Filter Performance by Timestamp

### Windows PowerShell

PowerShell's `curl` is an alias for `Invoke-WebRequest`. Use one of these:

**Option 1: Using Invoke-WebRequest (PowerShell native)**
```powershell
Invoke-WebRequest -Uri "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00" | Select-Object -ExpandProperty Content
```

**Option 2: Using curl.exe (if installed)**
```powershell
curl.exe "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00"
```

**Option 3: Using Python (most reliable)**
```powershell
python -c "import requests; import json; r = requests.get('http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00'); print(json.dumps(r.json(), indent=2))"
```

### Linux/Mac

```bash
curl "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00"
```

**Pretty JSON output:**
```bash
curl "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00" | python -m json.tool
```

## Find Your Resume Timestamp

Based on your trading journal, training started at:
- **First trade:** `2025-11-24T08:34:21.389377`
- **Last trade:** `2025-11-24T21:42:58.748538`

## Example Commands

### Get Performance Since Training Started

**PowerShell:**
```powershell
python -c "import requests; import json; r = requests.get('http://localhost:8200/api/monitoring/performance?since=2025-11-24T08:34:21'); print(json.dumps(r.json(), indent=2))"
```

**Linux/Mac:**
```bash
curl "http://localhost:8200/api/monitoring/performance?since=2025-11-24T08:34:21" | python -m json.tool
```

### Get Performance Since Checkpoint Resume (if you know the time)

If you resumed from checkpoint at a specific time, use that timestamp:

**PowerShell:**
```powershell
python -c "import requests; import json; r = requests.get('http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00'); print(json.dumps(r.json(), indent=2))"
```

### Get Performance for Last Hour

**PowerShell:**
```powershell
python -c "from datetime import datetime, timedelta; import requests; import json; since = (datetime.now() - timedelta(hours=1)).isoformat(); r = requests.get(f'http://localhost:8200/api/monitoring/performance?since={since}'); print(json.dumps(r.json(), indent=2))"
```

## Quick Reference

| Scenario | Command |
|----------|---------|
| **All trades** | `http://localhost:8200/api/monitoring/performance` |
| **Since timestamp** | `http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00` |
| **Last hour** | Use Python script above |
| **Since checkpoint resume** | Use timestamp when you resumed training |

## Response Format

The response includes:
- `filtered_since`: The timestamp used (or `null` if no filter)
- All performance metrics calculated only from filtered trades
- This shows performance of current training run only

