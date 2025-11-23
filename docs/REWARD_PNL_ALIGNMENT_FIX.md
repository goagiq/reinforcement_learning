# Reward-PnL Alignment Fix

**Date**: 2025-11-22  
**Issue**: Positive rewards but negative PnL - reward function was masking losses  
**Status**: ✅ **FIXED**

---

## Problem Identified

From the training dashboard:
- **Win Rate**: 36.7% (121 wins / 330 trades)
- **Current PnL**: -$1,562.83 (negative)
- **Mean PnL (Last 10)**: -$478.09 (negative)
- **BUT**: Rewards were positive (26.15 latest, 83.47 mean)

**Root Cause**: The reward function had multiple bonuses (action diversity, exploration, loss mitigation) that could make rewards positive even when PnL was negative. This misaligned the learning signal.

---

## Math Analysis

### Expected Value Calculation

With **36.7% win rate**:
- Required R:R for breakeven: `1 / (1 - 0.367) = 1.58`
- With **2.0 R:R**: `EV = (0.367 × 2.0) - (0.633 × 1.0) = 0.101` (10.1% positive)
- With **2.5 R:R**: `EV = (0.367 × 2.5) - (0.633 × 1.0) = 0.285` (28.5% positive)

**But PnL was negative**, which means:
1. Actual R:R was likely < 2.0 (not achieving target)
2. Commission costs were eating into profits
3. Reward function was masking the problem

---

## Fixes Applied

### 1. **Reward Function Alignment** (`src/trading_env.py`)

**Changes:**
- ✅ **PnL is PRIMARY signal** (90%+ of reward)
- ✅ **Bonuses only apply when PnL is positive** (don't mask losses)
- ✅ **Strong penalty when overall PnL is negative** (50% of negative PnL as penalty)
- ✅ **R:R penalty** - penalize if actual R:R < required R:R
- ✅ **Removed loss mitigation** - losses should be penalized fully
- ✅ **Reduced scaling** (5x → 3x) to keep rewards aligned with PnL

**Key Code:**
```python
# PRIMARY: PnL change is the main reward signal (90% weight)
reward = self.reward_config["pnl_weight"] * net_pnl_change

# CRITICAL: If overall PnL is negative, apply strong penalty
if total_pnl_normalized < -0.01:  # More than 1% down
    pnl_penalty = abs(total_pnl_normalized) * 0.5  # 50% of negative PnL as penalty
    reward -= pnl_penalty

# CRITICAL: If actual R:R < required R:R, apply penalty
if actual_rr_ratio > 0 and actual_rr_ratio < required_rr:
    rr_penalty = (required_rr - actual_rr_ratio) / required_rr * 0.1  # Up to 10% penalty
    reward -= rr_penalty
```

### 2. **Stricter R:R Enforcement** (`src/trading_env.py`)

**Changes:**
- ✅ **Require 10+ trades** before checking R:R (was 0)
- ✅ **10% buffer** - require R:R to be 10% above minimum (e.g., 2.5 × 1.1 = 2.75)
- ✅ **Reject trades** if actual R:R < required R:R with buffer

**Key Code:**
```python
if abs(position_change) > self.action_threshold and self.state.trades_count > 10:
    # Calculate actual R:R from recent trades
    avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades)
    avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades)
    risk_reward_ratio = avg_win / avg_loss
    
    # STRICT: Require 10% buffer above minimum
    required_rr_with_buffer = self.min_risk_reward_ratio * 1.1
    if risk_reward_ratio < required_rr_with_buffer:
        position_change = 0.0  # Reject trade
```

### 3. **Increased R:R Requirement** (`configs/train_config_adaptive.yaml`)

**Changes:**
- ✅ **Increased from 2.0 to 2.5** - with 36.7% win rate, need 2.5+ R:R for profitability

**Math:**
- With 2.5 R:R and 36.7% win rate: `EV = (0.367 × 2.5) - (0.633 × 1.0) = 0.285` (28.5% positive)
- This provides a safety margin above breakeven

---

## Expected Impact

### Before Fix:
- ✅ Positive rewards (masking negative PnL)
- ❌ Agent learns to optimize rewards, not PnL
- ❌ Poor R:R not properly penalized
- ❌ Losses masked by bonuses

### After Fix:
- ✅ Rewards align with actual PnL
- ✅ Agent learns to optimize PnL directly
- ✅ Poor R:R strongly penalized
- ✅ Losses fully penalized (no masking)

---

## Next Steps

1. **Monitor Training**: Watch for rewards to become negative when PnL is negative
2. **R:R Achievement**: Agent should achieve 2.5+ R:R to continue trading
3. **Win Rate**: If win rate improves, R:R requirement can be relaxed
4. **Adaptive Adjustment**: Adaptive trainer will adjust R:R threshold based on performance

---

## Configuration Changes

**File**: `configs/train_config_adaptive.yaml`

```yaml
reward:
  min_risk_reward_ratio: 2.5  # Increased from 2.0
```

**File**: `src/trading_env.py`

- Reward function: PnL-aligned (90%+ weight on PnL)
- R:R enforcement: Stricter (10% buffer, 10+ trades required)
- Bonuses: Only when PnL is positive

---

## Summary

✅ **Reward function now aligns with actual PnL**  
✅ **R:R requirement increased to 2.5**  
✅ **Stricter R:R enforcement with 10% buffer**  
✅ **Bonuses only apply when PnL is positive**  
✅ **Strong penalties for negative PnL and poor R:R**

The agent will now learn to optimize actual profitability, not just rewards.

