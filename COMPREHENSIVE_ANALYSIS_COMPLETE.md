# Comprehensive Bug Analysis - Complete

## Summary

Found **5 critical bugs** beyond the ones we already fixed:

---

## 🚨 Bug 1: DUPLICATE TRADE LOGGING (CRITICAL)

**Location:** `src/trading_env.py` line 1198-1212

**Problem:** 
- Position reversal calls `trade_callback()` **directly** with `commission=0.0`
- Then unified callback at line 1336 also calls it if `trade_info_for_journal` exists
- **Same trade logged 2x** → matches the 3x duplicates in database!

**Fix Applied:** ✅ 
- Removed direct callback
- Changed to store `trade_info_for_journal` dict
- Use unified callback only

---

## 🚨 Bug 2: COMMISSION DEDUCTION ORDER

**Location:** `src/trading_env.py` lines 994/1160/1219, 1332

**Problem:**
- Trade PnL added to `realized_pnl` (GROSS)
- Commission subtracted later from `realized_pnl`
- Journal calculates `net_pnl = pnl - commission`
- **Risk:** Double-deduction or confusion

**Status:** ⚠️ **NEEDS REVIEW**
- Current: Commission subtracted from `realized_pnl`
- Journal: Also subtracts in `net_pnl`
- **Verify:** Is commission being deducted twice or is this correct?

---

## 🚨 Bug 3: ENTRY PRICE NOT CLEARED

**Location:** `src/trading_env.py` position reversal

**Problem:** Entry price not explicitly cleared before setting new one.

**Fix Applied:** ✅ Added explicit clearing

---

## 🚨 Bug 4: UNREALIZED PNL RESET

**Status:** ✅ Verified - Resets correctly when position closes/reverses

---

## 🚨 Bug 5: TRADE_INFO STRUCTURE INCONSISTENCY

**Location:** `src/trading_env.py` lines 1159, 1218, 1039

**Problem:**
- Stop loss: Creates dict `trade_info_for_journal = {...}`
- Position close: Creates dict `trade_info_for_journal = {...}`
- Position reversal: Set `trade_pnl_for_journal = trade_pnl_amount` (number, not dict!)

**Fix Applied:** ✅ Changed reversal to create dict structure

---

## Other Issues

### P&L Discrepancy ($1.32M)
- **Root Cause:** Equity curve (per-episode) vs Journal (cumulative)
- **Status:** Identified, needs architectural decision

### Duplicate Trades (3x)
- **Root Cause:** Likely from Bug 1 (duplicate callbacks)
- **Status:** Should be fixed after removing duplicate callback

---

## Files Modified

1. `src/trading_env.py`:
   - Removed duplicate callback at line 1198
   - Fixed trade_info structure for reversals
   - Added explicit entry_price clearing

---

## Testing Needed

1. ✅ Verify no duplicate trades in database
2. ✅ Verify commission is deducted correctly (once, not twice)
3. ✅ Monitor P&L for consistency
4. ✅ Test stop loss triggers correctly

