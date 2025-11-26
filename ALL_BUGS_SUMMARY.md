# All Bugs Found - Summary

## Bugs Already Fixed Today

1. ✅ **Commission Double-Charging** - Fixed to charge only on entry
2. ✅ **Stop Loss Not Working** - Fixed to check loss from entry price
3. ✅ **Toggle Switch Filter** - Fixed with debug logging
4. ✅ **P&L Discrepancy** - Identified (equity curve per-episode vs journal cumulative)

## New Critical Bugs Found

### 🚨 Bug 1: DUPLICATE TRADE LOGGING

**Location:** `src/trading_env.py` line 1198-1212

**Problem:** Position reversal calls `trade_callback()` directly, then it's called again at line 1336.

**Impact:** Same trade logged **2x** to database - matches the duplicate trades we found in analysis!

**Status:** ⚠️ **PARTIALLY FIXED** - Changed to store trade_info instead of calling callback directly

---

### 🚨 Bug 2: COMMISSION DEDUCTION ORDER

**Location:** `src/trading_env.py` lines 994/1160/1219 (add PnL) and 1332 (subtract commission)

**Problem:** 
- Trade PnL added to `realized_pnl` (GROSS)
- Commission subtracted later (NET)
- But journal also calculates `net_pnl = pnl - commission`

**Risk:** Commission might be double-subtracted or not properly accounted for.

**Status:** ⚠️ **NEEDS REVIEW** - Commission is subtracted from `realized_pnl`, but journal also subtracts it in `net_pnl`

---

### 🚨 Bug 3: ENTRY PRICE CLEARING

**Location:** `src/trading_env.py` line 1213 (reversal)

**Problem:** Entry price not explicitly cleared before setting new one for reversal.

**Status:** ⚠️ **FIXED** - Added explicit clearing

---

### 🚨 Bug 4: UNREALIZED PNL RESET

**Location:** `src/trading_env.py` (position reversal)

**Problem:** Unrealized PnL might not be reset when position reverses.

**Status:** ⚠️ **NEEDS VERIFICATION** - Should reset when old position is realized

---

### 🚨 Bug 5: COMMISSION IN JOURNAL

**Location:** `src/trading_journal.py` line 299

**Problem:** Journal calculates `net_pnl = pnl - commission`, but `realized_pnl` also has commission subtracted.

**Question:** Is commission being subtracted twice?

---

## Recommended Actions

1. **Verify duplicate logging is fixed** (removed direct callback at line 1198)
2. **Review commission flow** - ensure it's only subtracted once
3. **Test with actual trades** - verify no duplicates appear
4. **Monitor P&L consistency** - ensure equity matches journal

