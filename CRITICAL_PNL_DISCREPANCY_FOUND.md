# CRITICAL P&L DISCREPANCY FOUND

## Root Cause Identified

**MASSIVE DISCREPANCY: $1,320,713.20**

### The Problem:

1. **Equity Curve:**
   - Shows **PER-EPISODE** PnL (resets each episode)
   - Latest: $99,964.53 (only Episode 139, Step 50)
   - Uses `self.state.total_pnl` which **resets to 0** at episode start
   - This is **not cumulative** across all episodes

2. **Trading Journal (Performance Monitor):**
   - Shows **CUMULATIVE** PnL (sum of ALL trades from ALL episodes)
   - Total Net PnL: **-$1,320,748.68**
   - Expected Equity: -$1,220,748.68
   - This is **cumulative** across all episodes

3. **The Discrepancy:**
   - Equity Curve: $99,964.53 (current episode only)
   - Expected (from journal): -$1,220,748.68 (all episodes)
   - **Difference: $1,320,713.20!**

### Why This Happens:

1. **Equity calculation in `trading_env.py`:**
   ```python
   current_equity = self.initial_capital + self.state.total_pnl
   ```
   - `self.state.total_pnl` **resets to 0** at episode start (line 914)
   - This is **per-episode**, not cumulative

2. **Trade logging:**
   - Every trade is logged to database with `net_pnl`
   - Database **sums all trades** from all episodes
   - This is **cumulative**

3. **Performance Monitor (`api_server.py`):**
   - Reads from trading journal database
   - Uses `SUM(net_pnl)` from ALL trades
   - This is **cumulative**

### Additional Issues Found:

1. **Duplicate Trades:**
   - Some trades logged 3 times (same timestamp/episode/step)
   - May be double-counting losses

2. **Commission:**
   - $30 flat rate per trade (not percentage)
   - Total commission: $1.31M
   - This is still too high even after the fix

### The Fix:

**Option 1: Make Performance Monitor show current session only**
- Filter trades by `training_start_timestamp`
- This would match the equity curve (current episode only)

**Option 2: Make Equity Curve cumulative**
- Track cumulative PnL across all episodes
- Store it in a persistent variable that doesn't reset
- This would match the trading journal

**Option 3: Use Trading Journal for both**
- Calculate equity from cumulative sum of trades
- Equity = initial_capital + SUM(net_pnl)
- This is the most accurate

### Recommended Fix:

**Use Trading Journal for both equity and P&L:**
- Performance Monitor should calculate equity as: `initial_capital + SUM(net_pnl from all trades)`
- This ensures consistency between equity curve and P&L
- The equity curve in the database should also be cumulative, not per-episode

## Next Steps:

1. Fix duplicate trade logging (prevent double-counting)
2. Fix equity calculation to be cumulative (not per-episode)
3. Ensure Performance Monitor uses cumulative P&L from database
4. Verify commission calculation (should be ~$13/trade after fix)

