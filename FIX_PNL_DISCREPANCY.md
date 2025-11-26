# Fix P&L Discrepancy Between Training Progress and Performance Monitoring

## Root Cause

**Training Progress and Performance Monitoring use different data sources:**

1. **Training Progress:**
   - Uses `trainer.env.state.total_pnl` (in-memory, current episode)
   - Resets each episode
   - Shows only current episode/session PnL

2. **Performance Monitoring:**
   - Uses database sum of `net_pnl` (persistent, cumulative)
   - Filters by timestamp (current session)
   - Shows all trades in current session

**The problem:** These should match but they don't!

## Why They Don't Match

1. **Training Progress:** Shows positive because current episode is profitable
2. **Performance Monitoring:** Shows negative because it includes ALL trades from current session (many losing trades)

## Solution

**Make both panels use the same data source and calculation:**

### Option 1: Make Training Progress use database (recommended)
- Training Progress should read from database, filtered by current session
- This ensures consistency across panels

### Option 2: Make Performance Monitoring use environment state
- Performance Monitoring should read from `trainer.env.state.total_pnl`
- But this only shows current episode, not all trades

### Option 3: Add session filtering to Training Progress
- Training Progress should show current session (all episodes since training start)
- Not just current episode

## Recommended Fix

**Update Training Progress to show current session PnL (not just current episode):**

1. Read PnL from database, filtered by `training_start_timestamp`
2. Match the same filtering logic as Performance Monitoring
3. Show both:
   - Current Episode PnL (from environment state)
   - Current Session PnL (from database, matches Performance Monitoring)

This way:
- Both panels show the same "Current Session" PnL
- Training Progress also shows "Current Episode" for detailed view
- No arithmetic discrepancy!

