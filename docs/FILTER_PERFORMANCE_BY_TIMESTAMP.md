# Filter Performance by Timestamp

## Get Timestamp When Training Resumed

First, you need to find when training resumed from checkpoint. Check the backend console logs for a message like:
```
📂 Resuming from checkpoint: models/checkpoint_1000000.pt
```

Or check the trading journal for the first trade after resume.

## Curl Command

### Basic Format

```bash
curl "http://localhost:8200/api/monitoring/performance?since=YYYY-MM-DDTHH:MM:SS"
```

### Example (Replace with your actual timestamp)

```bash
# Example: Filter trades since November 24, 2025 at 7:00 PM
curl "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00"
```

### With Pretty JSON Output

```bash
curl "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00" | python -m json.tool
```

### Windows PowerShell

```powershell
# Basic
Invoke-WebRequest -Uri "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00" | Select-Object -ExpandProperty Content

# Pretty JSON
Invoke-WebRequest -Uri "http://localhost:8200/api/monitoring/performance?since=2025-11-24T19:00:00" | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## Find Your Resume Timestamp

### Option 1: Check Backend Logs
Look for the timestamp when you see:
```
📂 Resuming from checkpoint: models/checkpoint_1000000.pt
```

### Option 2: Query Trading Journal

```bash
# Get first trade timestamp (oldest trade = when training started)
curl "http://localhost:8200/api/journal/trades?limit=1&offset=0" | python -m json.tool
```

### Option 3: Use SQLite Directly

```bash
python -c "import sqlite3; conn = sqlite3.connect('logs/trading_journal.db'); cursor = conn.cursor(); cursor.execute('SELECT MIN(timestamp) FROM trades'); print(cursor.fetchone()[0]); conn.close()"
```

## Example: Get Performance Since Checkpoint Resume

```bash
# Step 1: Find when training resumed (get oldest trade timestamp)
OLDEST_TRADE=$(python -c "import sqlite3; conn = sqlite3.connect('logs/trading_journal.db'); cursor = conn.cursor(); cursor.execute('SELECT MIN(timestamp) FROM trades'); print(cursor.fetchone()[0]); conn.close()")

# Step 2: Use that timestamp to filter
curl "http://localhost:8200/api/monitoring/performance?since=${OLDEST_TRADE}" | python -m json.tool
```

## Timestamp Format

- **Format:** ISO 8601: `YYYY-MM-DDTHH:MM:SS`
- **Example:** `2025-11-24T19:00:00`
- **Timezone:** Uses local timezone (or UTC if specified)

## Response

The response will include:
- `filtered_since`: The timestamp used for filtering
- All metrics calculated only from trades after that timestamp
- This gives you performance of the current training run only

## Quick Test

```bash
# Get current time minus 1 hour (to see last hour's performance)
curl "http://localhost:8200/api/monitoring/performance?since=$(python -c "from datetime import datetime, timedelta; print((datetime.now() - timedelta(hours=1)).isoformat())")" | python -m json.tool
```

