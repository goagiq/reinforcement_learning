# Performance Monitoring After Checkpoint Resume

## Understanding the Behavior

When you resume training from a checkpoint, the **Performance Monitoring dashboard shows cumulative statistics** from the trading journal, which includes:

- ✅ **All trades from previous training runs** (before checkpoint)
- ✅ **All trades from current training run** (after checkpoint resume)

This is **expected behavior** - the trading journal is a persistent database that accumulates all trades over time.

## Why This Happens

The trading journal (`logs/trading_journal.db`) is designed to:
1. **Persist across training sessions** - Never loses trade data
2. **Track all trades** - Complete historical record
3. **Enable analysis** - Compare performance over time

When you resume from checkpoint, new trades are **added** to the journal, not replacing old ones.

## Solutions

### Option 1: Filter by Timestamp (Recommended)

The performance endpoint now supports filtering by timestamp:

```bash
# Get performance metrics since a specific time
GET /api/monitoring/performance?since=2025-11-24T19:00:00
```

**How to use:**
1. Note the timestamp when you resumed training (check backend logs)
2. Use that timestamp in the API call
3. Frontend will show only trades after that time

**Example:**
- Resumed training at: `2025-11-24T19:00:00`
- API call: `/api/monitoring/performance?since=2025-11-24T19:00:00`
- Result: Only shows trades from the current training run

### Option 2: View All Trades (Current Behavior)

By default, the dashboard shows **all trades** (cumulative):
- Total PnL across all training
- Overall win rate
- Complete trading history

This is useful for:
- Seeing overall system performance
- Tracking long-term trends
- Comparing different training runs

### Option 3: Clear Journal (Not Recommended)

You could clear the journal before resuming, but this:
- ❌ Loses all historical data
- ❌ Makes it hard to compare performance
- ❌ Not recommended unless you want a fresh start

## Frontend Integration

The frontend can be updated to:
1. **Auto-detect checkpoint resume** - Track when training resumed
2. **Show toggle** - "All Time" vs "Since Resume"
3. **Display filter status** - Show if filtering is active

## Example: Filtering After Checkpoint Resume

**Scenario:**
- Checkpoint: `checkpoint_1000000.pt` (resumed at 1M timesteps)
- Resume time: `2025-11-24T19:00:00`
- Old trades: 3,109 trades (before resume)
- New trades: 50 trades (after resume)

**Without filter:**
- Shows all 3,159 trades
- Cumulative PnL: Includes old + new trades
- Win rate: Overall across all training

**With filter (`since=2025-11-24T19:00:00`):**
- Shows only 50 new trades
- PnL: Only from current training run
- Win rate: Only from current training run

## Recommendation

**For now:** The old values are **OK** - they represent cumulative performance.

**To see only new trades:** Use the `since` parameter with the timestamp when you resumed training.

**Future enhancement:** Add a frontend toggle to switch between "All Time" and "Since Resume" views automatically.

## Technical Details

The trading journal stores:
- `timestamp`: When the trade occurred
- `episode`: Episode number
- `step`: Step within episode

You can filter by:
- **Timestamp**: `WHERE timestamp >= ?`
- **Episode**: `WHERE episode >= ?` (if you track resume episode)

The performance endpoint now supports timestamp filtering via the `since` query parameter.

