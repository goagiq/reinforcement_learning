# Final Bug Summary - All Issues Found

## ✅ Bugs Fixed Today

1. **Commission Double-Charging** - Now charges only on entry (not exit)
2. **Stop Loss Logic** - Now checks loss from entry price (not previous step)
3. **Toggle Switch** - Added debug logging for timestamp filtering

## 🚨 Critical Bugs Found (Need Fix)

### Bug 1: Duplicate Trade Logging

**Location:** `src/trading_env.py`

**Issue:** Position reversal (line 1198-1212) calls `trade_callback()` directly with `commission=0.0`, potentially causing duplicate logging.

**Status:** ⚠️ Partially addressed - changed to store trade_info instead of calling directly

### Bug 2: Commission Deduction Order

**Issue:** Commission is subtracted from `realized_pnl` AFTER trade PnL is added. Journal also calculates `net_pnl = pnl - commission`. Could lead to double-deduction or confusion.

**Status:** ⚠️ Needs review - verify commission is only subtracted once

### Bug 3: Entry Price Clearing

**Status:** ✅ Fixed - Added explicit clearing for reversals

---

## 🔍 Other Issues Identified

### Issue A: P&L Discrepancy ($1.32M difference)

**Root Cause:**
- Equity curve uses **per-episode** PnL (resets each episode)
- Trading journal uses **cumulative** PnL (all trades)

**Solution:** Use cumulative PnL for both, or make it clear they show different scopes.

### Issue B: Duplicate Trades in Database

**Root Cause:** Possibly from multiple callbacks or same trade being logged multiple times.

**Status:** ⚠️ Investigating

---

## Recommendations

1. **Test duplicate logging fix** - Verify no more 3x duplicates
2. **Review commission flow** - Ensure single deduction
3. **Monitor P&L** - Watch for consistency improvements
4. **Clear documentation** - Document per-episode vs cumulative P&L

