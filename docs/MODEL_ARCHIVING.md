# Model Archiving System

## Overview

The training system now automatically archives existing model files before starting fresh training. This prevents accidentally reusing old (potentially problematic) models.

## How It Works

### Automatic Archiving

When you start training **from scratch** (no checkpoint, no transfer learning), the system will:

1. **Scan** the `models/` directory for all `.pt` files
2. **Create** a timestamped archive folder: `models/Archive/archive_YYYYMMDD_HHMMSS/`
3. **Move** all model files to the archive folder
4. **Start** fresh training with a clean `models/` directory

### Archive Structure

```
models/
├── Archive/
│   ├── archive_20241122_143022/
│   │   ├── checkpoint_10000.pt
│   │   ├── checkpoint_20000.pt
│   │   ├── ...
│   │   ├── checkpoint_8000000.pt
│   │   ├── best_model.pt
│   │   └── final_model.pt
│   └── archive_20241122_150145/
│       └── ... (next training session)
├── checkpoint_10000.pt  (new training)
├── checkpoint_20000.pt  (new training)
└── best_model.pt        (new training)
```

## When Archiving Happens

Archiving is triggered when:
- ✅ **Starting fresh training** (no checkpoint provided)
- ✅ **Transfer learning is disabled** (`transfer_learning: false`)

Archiving is **NOT** triggered when:
- ❌ Resuming from a checkpoint (checkpoint_path provided)
- ❌ Using transfer learning (`transfer_learning: true`)

## Benefits

1. **Prevents Accidental Reuse**: Old models won't be accidentally loaded
2. **Clean Slate**: Each training session starts with empty models directory
3. **Preserves History**: All old models are safely archived with timestamps
4. **Easy Recovery**: Archived models can be manually restored if needed

## Manual Archive Management

### View Archived Models

```bash
# List all archive folders
ls models/Archive/

# View contents of specific archive
ls models/Archive/archive_20241122_143022/
```

### Restore Archived Models

If you need to restore an archived model:

```bash
# Copy from archive
cp models/Archive/archive_20241122_143022/best_model.pt models/

# Or move back
mv models/Archive/archive_20241122_143022/best_model.pt models/
```

### Clean Up Old Archives

To free up disk space, you can delete old archive folders:

```bash
# Remove specific archive
rm -rf models/Archive/archive_20241122_143022/

# Or keep only recent N archives (manual cleanup)
```

## Configuration

The archiving behavior is automatic and doesn't require configuration. However, you can control it indirectly:

```yaml
training:
  transfer_learning: false  # Set to false to trigger archiving on fresh start
```

## Example Output

When archiving is triggered, you'll see:

```
======================================================================
ARCHIVING EXISTING MODELS
======================================================================
Archive folder: models/Archive/archive_20241122_143022
Found 802 model file(s) to archive:
  [OK] Archived: checkpoint_10000.pt
  [OK] Archived: checkpoint_20000.pt
  ...
  [OK] Archived: checkpoint_8000000.pt
  [OK] Archived: best_model.pt
  [OK] Archived: final_model.pt

Archived 802 model file(s) to: models/Archive/archive_20241122_143022
======================================================================
```

## Notes

- **Archive folders are timestamped**: Each training session creates a new archive folder
- **Files are moved, not copied**: Original files are moved to archive (saves disk space)
- **Archive folder is created automatically**: No manual setup needed
- **Safe operation**: If archiving fails for a file, it continues with others

## Troubleshooting

### Issue: "Failed to archive [file]"

**Cause**: File might be locked or in use

**Solution**: 
- Ensure no other process is using the model files
- Check file permissions
- Manually move the file if needed

### Issue: Archive folder not created

**Cause**: Permission issues

**Solution**:
- Check write permissions on `models/` directory
- Ensure `models/Archive/` can be created

### Issue: Want to disable archiving

**Solution**: 
- Provide a checkpoint path (even if dummy) to skip archiving
- Or modify the code to add a config flag (future enhancement)

