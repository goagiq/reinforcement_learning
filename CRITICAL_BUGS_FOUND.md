# Critical Bugs Found - Comprehensive Analysis

## 🚨 Bug 1: DUPLICATE TRADE LOGGING (CRITICAL)

**Location:** `src/trading_env.py`

**Problem:** Trade callback is called **MULTIPLE TIMES** for the same trade:

1. **Position Reversal** (line 1198-1212): Calls `trade_callback()` **DIRECTLY** with `commission=0.0`
2. **Final Callback** (line 1336-1355): Also calls `trade_callback()` if `trade_info_for_journal` exists

**Impact:**
- **Same trade logged 2x** (once at line 1198, once at line 1336)
- Causes duplicate trades in database (matches the 3x duplicates we found!)
- Inflated trade counts
- Incorrect P&L (if journal sums trades)

**Fix:** Remove the direct callback at line 1198. Use ONLY the unified callback at line 1336.

---

## 🚨 Bug 2: COMMISSION DEDUCTION ORDER (CRITICAL)

**Location:** `src/trading_env.py` lines 994, 1160, 1219, 1332

**Problem:** Commission is subtracted AFTER trade PnL is added:

1. Line 994/1160/1219: `realized_pnl += trade_pnl_amount` (GROSS PnL, no commission)
2. Line 1332: `realized_pnl -= commission_cost` (subtract commission)

**But:** The journal calculates `net_pnl = pnl - commission` (line 299 in `trading_journal.py`)

**Risk:** 
- Commission might be subtracted twice (once from `realized_pnl`, once in journal's `net_pnl`)
- OR commission might not be properly accounted for

**Fix:** Either:
- Option A: Subtract commission from `trade_pnl_amount` BEFORE adding to `realized_pnl`
- Option B: Don't subtract commission from `realized_pnl` (let journal handle via `net_pnl`)

---

## 🚨 Bug 3: COMMISSION FOR REVERSALS

**Location:** `src/trading_env.py` line 1207

**Problem:** Position reversal callback passes `commission=0.0` with comment "Commission calculated later"

**Current Flow:**
- Reversal callback called at line 1198 with `commission=0.0`
- Then commission calculated at line 1327
- But callback already happened!

**Fix:** Remove direct callback at line 1198, use unified callback at line 1336.

---

## 🚨 Bug 4: ENTRY PRICE NOT CLEARED FOR REVERSALS

**Location:** `src/trading_env.py` lines 1154-1270

**Problem:** When position reverses:
- Old position's entry_price is NOT explicitly cleared
- New entry_price is set at line 1313
- But old entry_price could persist if reversal logic doesn't clear it

**Impact:** Stale entry_price could cause incorrect PnL calculations.

**Fix:** Explicitly clear `entry_price` before setting new one for reversal.

---

## 🚨 Bug 5: UNREALIZED PNL NOT RESET FOR NEW POSITION

**Location:** `src/trading_env.py` lines 1154-1313

**Problem:** When position reverses:
- Old position's unrealized PnL is realized (line 1160)
- But `unrealized_pnl` variable is NOT reset for the new position
- It's only recalculated later at line 1359

**Impact:** Could cause incorrect total_pnl calculation temporarily.

**Fix:** Reset `unrealized_pnl = 0.0` when position reverses, before recalculating for new position.

---

## Summary of Fixes Needed

1. ✅ **Remove duplicate callback** at line 1198 (position reversal)
2. ✅ **Fix commission deduction order** - subtract before adding to realized_pnl
3. ✅ **Clear entry_price** explicitly for reversals
4. ✅ **Reset unrealized_pnl** for new positions after reversal
5. ✅ **Verify commission calculation** for reversals

---

## Files to Modify

1. `src/trading_env.py`:
   - Remove callback at line 1198
   - Fix commission deduction order
   - Add entry_price clearing for reversals
   - Reset unrealized_pnl properly

