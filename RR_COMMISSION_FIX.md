# R:R Ratio Commission Fix

## Problem Identified

The current R:R setting (1.5:1) **does NOT adequately account for commission costs**.

### Analysis Results:
- **Current Net R:R: 0.71:1** (terrible - below break-even of 1.21:1)
- **Gross R:R: 1.20:1** (before commission)
- **Commission Impact: $28.51 per trade (31.37% of net win)**
- **To achieve 1.5:1 net R:R, need 2.51:1 gross R:R!**

### Root Cause:
- Commission is deducted from PnL, so R:R is calculated from net values
- But commission eats into profits significantly (31% of net win)
- Current 1.5:1 requirement is too low to be profitable with high commission costs
- System is losing money rapidly because R:R is only 0.71:1

## Fix Applied

### 1. Increased Minimum R:R Requirement
- **Before:** `min_risk_reward_ratio: 1.5`
- **After:** `min_risk_reward_ratio: 2.0`
- **Reason:** Accounts for commission costs - need higher net R:R to be profitable

### 2. Tightened R:R Enforcement Floor
- **Before:** `min_acceptable_rr_floor = 0.7` (only rejects catastrophic trades)
- **After:** `min_acceptable_rr_floor = 1.0` (rejects trades below break-even)
- **Reason:** Prevents further losses from poor R:R trades

### 3. Updated DecisionGate R:R Requirement
- **Before:** `min_risk_reward_ratio: 1.5`
- **After:** `min_risk_reward_ratio: 2.0`
- **Reason:** Consistent with reward config

## Expected Impact

### With 2.0:1 Net R:R:
- Average Win: $255.86 (vs current $90.89)
- Average Loss: $127.93 (unchanged)
- Expected Value: (0.4527 × $255.86) - (0.5473 × $127.93) = **$115.78 - $70.02 = +$45.76 per trade**
- **Result: PROFITABLE!** (vs current -$28.88 per trade)

### Break-Even Analysis:
- With 45% win rate, need minimum 1.21:1 to break even
- But with commission at 31% of net win, actual break-even is higher (~1.8:1)
- Setting 2.0:1 provides safety margin for profitability

## Files Modified

1. `configs/train_config_adaptive.yaml`:
   - `environment.reward.min_risk_reward_ratio: 1.5 → 2.0`
   - `decision_gate.quality_scorer.min_risk_reward_ratio: 1.5 → 2.0`

2. `src/trading_env.py`:
   - `min_acceptable_rr_floor: 0.7 → 1.0`

## Next Steps

1. **Monitor R:R ratio** - Should improve from 0.71:1 toward 2.0:1
2. **Watch for reduced trading frequency** - Higher R:R requirement may reduce trades initially
3. **Check profitability** - Expected to turn profitable as R:R improves
4. **Consider commission reduction** - If trades become too infrequent, may need to reduce commission_rate from 0.0003

## Note

The reward function already has strong R:R penalties (up to 30% for poor R:R) and bonuses for good R:R (>1.5:1). With the higher requirement, the agent should learn to:
- Let winners run longer (to achieve 2.0:1 R:R)
- Cut losses faster
- Only take high-quality trades with good risk/reward

