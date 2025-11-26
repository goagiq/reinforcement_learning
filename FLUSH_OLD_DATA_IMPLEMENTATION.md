# Flush Old Training Data Implementation

## Overview
Added an option to clear old training data (databases, caches) when starting fresh training. This ensures that metrics and dashboards show only data from the new training session, avoiding confusion from historical data.

## Changes Made

### 1. Backend Changes

#### `src/api_server.py`
- **Added `flush_old_data` parameter** to `TrainingRequest` model (line 138)
  ```python
  flush_old_data: bool = False  # Clear trading journal and caches when starting fresh training
  ```
- **Integrated flush logic** in `/api/training/start` endpoint (lines 1038-1055)
  - Only triggers when `flush_old_data=True` AND no checkpoint is specified (fresh start only)
  - Calls `clear_all_training_data()` to archive and clear:
    - Trading journal database (archived to `logs/trading_journal_archive/`)
    - Cache files (`logs/known_files_cache.json`, etc.)
    - Optionally processed data directory (currently disabled)

#### `src/clear_training_data.py` (NEW FILE)
Utility module providing functions to clear training data:

- **`clear_trading_journal(db_path, archive=True)`**
  - Counts existing trades before clearing
  - Archives database to `logs/trading_journal_archive/trading_journal_backup_TIMESTAMP.db` if `archive=True`
  - Deletes the original database file
  
- **`clear_caches(cache_paths=None)`**
  - Clears cache files (default: `logs/known_files_cache.json`)
  - Optionally clears processed data directory contents
  
- **`clear_all_training_data(archive_db=True, clear_caches=True, clear_processed=False)`**
  - Main function that orchestrates all clearing operations
  - Returns comprehensive status dictionary

### 2. Frontend Changes

#### `frontend/src/components/TrainingPanel.jsx`
- **Added state variable** for flush option (line 145)
  ```javascript
  const [flushOldData, setFlushOldData] = useState(false)
  ```
- **Added checkbox UI** (lines 1370-1391)
  - Only visible when `selectedCheckpoint === 'none'` (fresh start)
  - Styled with yellow warning background
  - Includes detailed explanation of what will be cleared
  - Shows that old data will be backed up before clearing
  
- **Included in training request** (lines 935-938)
  - Sends `flush_old_data: true/false` in request payload
  - Only included when starting fresh (no checkpoint)

## How It Works

1. **User selects "Start Fresh Training"** in the Training tab
2. **Checkbox appears** below the checkpoint selection
3. **User can check "Clear old training data"** to enable the flush
4. **When training starts**:
   - Backend checks if `flush_old_data=True` and no checkpoint
   - Archives trading journal database (if it exists)
   - Clears cache files
   - Starts training with clean slate
5. **Metrics dashboard** will only show trades from the new training session

## Safety Features

- **Automatic Backup**: Trading journal is archived before clearing (not deleted)
- **Fresh Start Only**: Flush only works when no checkpoint is specified (prevents accidental data loss when resuming)
- **Error Handling**: If flush fails, training still proceeds (logged as warning)
- **Explicit Opt-in**: User must explicitly check the checkbox (default: `false`)

## Files Modified

1. `src/api_server.py` - Added parameter and flush logic
2. `src/clear_training_data.py` - NEW utility module
3. `frontend/src/components/TrainingPanel.jsx` - Added UI checkbox

## Usage

1. Go to Training tab
2. Select "Start Fresh Training" from checkpoint dropdown
3. Check the "🗑️ Clear old training data (database, caches)" checkbox
4. Click "Start Training"
5. Old data will be archived and cleared before training begins

## Backup Location

Old trading journal databases are backed up to:
```
logs/trading_journal_archive/trading_journal_backup_YYYYMMDD_HHMMSS.db
```

This allows you to recover old trade data if needed.

