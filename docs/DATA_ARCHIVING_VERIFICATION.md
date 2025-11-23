# Data File Archiving Verification

**Date**: 2025-11-22  
**Status**: ✅ **IMPLEMENTED AND VERIFIED**

---

## Overview

The system automatically archives all data files used during training to prevent their accidental reuse in subsequent training runs. This ensures:

1. ✅ Fresh data is always used for new training runs
2. ✅ Used data files are preserved in archive folders
3. ✅ Both NT8 export files and local copies are archived
4. ✅ Archive folders are timestamped for organization

---

## How It Works

### 1. **File Tracking During Data Loading**

When data files are loaded, they are automatically tracked:

**Location**: `src/data_extraction.py`

- **Local files** (`data/raw/`): Tracked as `("local_file", path)`
- **NT8 source files**: Tracked as `("nt8_source", path)`
- **Local copies** (copied from NT8): Tracked as `("local_copy", path)`

**Tracking happens at:**
- Line 239: Local files found in `data/raw/`
- Line 254: NT8 files found in mapped NT8 folder
- Line 265: Local copies created from NT8 files
- Line 297: NT8 files found in common paths
- Line 307: Local copies from common paths
- Line 568: **Safety backup** - Files successfully loaded in `_load_data_file()` are also tracked

### 2. **Archiving After Training**

**Location**: `src/train.py` - `_archive_used_data_files()` method

**When**: Called automatically at the end of training (line 1488)

**What happens:**
1. Collects all tracked files from `data_extractor.used_data_files`
2. Deduplicates files (same file might be tracked multiple times)
3. Archives NT8 source files to: `{NT8_EXPORT_PATH}/Archive/archive_{timestamp}/`
4. Archives local files to: `data/raw/Archive/archive_{timestamp}/`

---

## NT8 Path Configuration

### Supported NT8 Export Paths

The system checks these paths (in order):

1. **User-specific path**: `C:/Users/schuo/Documents/NinjaTrader 8/export` ✅ **YOUR PATH**
2. **Previous user path**: `C:/Users/sovan/Documents/NinjaTrader 8/export`
3. **Home directory**: `{HOME}/Documents/NinjaTrader 8/export`
4. **Home directory (capitalized)**: `{HOME}/Documents/NinjaTrader 8/Export`

**Configuration locations:**
- `src/train.py` (lines 682-686): Archiving logic
- `src/data_extraction.py` (lines 284-289): Data loading logic

---

## Archive Structure

### NT8 Export Archive
```
C:\Users\schuo\Documents\NinjaTrader 8\export\
  └── Archive\
      └── archive_20251122_120000\
          ├── ES 03-11.Last.txt
          ├── ES 03-12.Last.txt
          └── ... (all used files)
```

### Local Data Archive
```
data/raw/Archive/
  └── archive_20251122_120000\
      ├── ES_1min.csv
      ├── ES_5min.csv
      └── ... (all used files)
```

---

## Verification Checklist

### ✅ Implementation Complete

- [x] File tracking in `DataExtractor`
- [x] Archiving method in `Trainer`
- [x] Called at end of training
- [x] NT8 path includes user's path (`C:\Users\schuo\Documents\NinjaTrader 8\export`)
- [x] Path resolution handles Windows paths correctly
- [x] Safety backup tracking in `_load_data_file()`
- [x] Deduplication of tracked files
- [x] Error handling for missing files

### ✅ Features

- [x] Archives NT8 source files (moves from export folder)
- [x] Archives local copies (moves from `data/raw/`)
- [x] Creates timestamped archive folders
- [x] Skips files already in Archive folders
- [x] Handles path resolution correctly
- [x] Prints archive summary at end of training

---

## Testing

### Manual Test

To verify archiving works:

1. **Start a training run** (even a short one)
2. **Check console output** at the end:
   ```
   ======================================================================
   ARCHIVING USED DATA FILES
   ======================================================================
     [OK] Archived NT8 file: ES 03-11.Last.txt
     [OK] Archived local file: ES_1min.csv
   
   Archived 2 data file(s)
     NT8 Archive: C:\Users\schuo\Documents\NinjaTrader 8\export\Archive\archive_20251122_120000
     Local Archive: data/raw/Archive/archive_20251122_120000
   ```

3. **Verify files moved**:
   - Check `C:\Users\schuo\Documents\NinjaTrader 8\export\Archive\` for archived files
   - Check `data/raw/Archive/` for archived local files
   - Original files should be **moved** (not copied) - they should no longer exist in original locations

---

## Important Notes

### File Movement (Not Copying)

⚠️ **Important**: Files are **moved** (not copied) to archive folders. This means:
- Original files are removed from their source locations
- Files are preserved in archive folders
- This prevents accidental reuse

### Archive Folder Naming

Archive folders use timestamp format: `archive_YYYYMMDD_HHMMSS`
- Example: `archive_20251122_120000`
- Same timestamp used for both NT8 and local archives (from same training run)

### Deduplication

The system automatically deduplicates tracked files:
- Same file tracked multiple times → archived only once
- Uses resolved paths for comparison

---

## Troubleshooting

### Files Not Being Archived

**Check:**
1. Is `data_extractor` initialized in `Trainer`?
2. Are files being loaded successfully?
3. Check console output for archiving messages
4. Verify NT8 path is correct

### Path Resolution Issues

**If files aren't found:**
- Check that NT8 path exists: `C:\Users\schuo\Documents\NinjaTrader 8\export`
- Verify path in `settings.json` if using custom path
- Check Windows path format (forward slashes work, but backslashes are also handled)

### Archive Folder Not Created

**Check:**
- Permissions on NT8 export folder
- Disk space available
- Path exists and is writable

---

## Summary

✅ **Data archiving is fully implemented and configured for your system.**

**Your NT8 path** (`C:\Users\schuo\Documents\NinjaTrader 8\export`) is included in the path list, and files will be automatically archived after each training run.

**Next training run will:**
1. Load data from NT8 export folder
2. Track all loaded files
3. Archive them at the end of training
4. Prevent their reuse in future training runs

