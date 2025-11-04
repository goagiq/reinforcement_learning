# Reward Optimization Analysis - Current Status vs Expected

## 📊 Current Training Status (480K timesteps / 48%)

**Metrics:**
- Mean Reward: **-40.65** (should be positive by now)
- Best Reward: **-19.48** (still negative)
- Episodes Completed: **2** (very few - episodes are long: 10K steps)
- Mean Episode Length: **9,980 steps**

**Expected at 48%:**
- Mean Reward: Should be **positive (0.5-1.0+)**
- Best Reward: Should be **positive**
- Episodes: Should have **more completed episodes**

---

## 🔍 Root Cause Analysis

### Problem 1: Reward Function Still Too Punitive

**Current Reward Formula:**
```python
reward = (
    pnl_weight * pnl_change                                    # 1.0 * pnl_change
    - risk_penalty * 0.1 * drawdown                           # -0.5 * 0.1 * drawdown = -0.05 * drawdown
    - drawdown_penalty * 0.1 * max(0, max_drawdown - 0.15)   # -0.3 * 0.1 * max(0, DD - 0.15) = -0.03 * excess_DD
    - holding_cost (0.001 * transaction_cost)                # Very small
)
if pnl_change > 0:
    reward += abs(pnl_change) * 0.1                          # Bonus for profits
reward *= 10.0                                                # Scale by 10
```

**Issues:**
1. **Penalties apply even when PnL is small/zero** - If agent makes no trades or small trades, penalties dominate
2. **Drawdown penalty accumulates** - Even small drawdowns get penalized
3. **No reward for exploration** - Agent learns to avoid trading (which gives slightly negative rewards)
4. **Transaction costs accumulate** - Even minimal holding cost adds up over 10K steps

### Problem 2: Agent Not Trading Enough

**Symptoms:**
- Only 2 episodes in 480K timesteps
- Mean reward consistently negative
- Agent may be learning to **avoid trading** (which gives zero or slightly negative rewards)

**Why this happens:**
- Random exploration early → negative rewards
- Agent learns: "No trading = small negative reward"
- Agent learns: "Trading = larger negative reward (from losses)"
- Agent converges to: "Avoid trading" strategy

### Problem 3: Reward Scaling May Still Be Too Aggressive

**Current:** 10x scaling
- If pnl_change = 0.001 (0.1% profit), reward = 0.01
- If drawdown = 0.05 (5%), penalty = -0.05 * 0.05 * 10 = -0.025
- **Net reward: -0.015** (negative even with profit!)

---

## 🎯 Recommended Optimizations

### Optimization 1: Reduce Penalties Further

**Change:**
- Reduce risk_penalty coefficient from 0.5 to **0.2**
- Reduce drawdown_penalty coefficient from 0.3 to **0.1**
- Only apply drawdown penalty if DD > **20%** (was 15%)

### Optimization 2: Add Exploration Bonus

**Add reward for trading activity:**
- Small bonus for opening positions (encourages exploration)
- Bonus for closing profitable trades
- Reduces penalty for unprofitable trades (learning opportunity)

### Optimization 3: Optimize Reward Scaling

**Change scaling from 10x to 5x:**
- Less aggressive scaling
- Allows more granular learning
- Prevents penalties from dominating

### Optimization 4: Increase Learning Rate Slightly

**Current:** 0.0003
**Recommended:** 0.0005
- Faster learning
- More responsive to positive rewards

### Optimization 5: Adjust Entropy Coefficient

**Current:** 0.01
**Recommended:** 0.02-0.03
- More exploration
- Prevents premature convergence to "no trading" strategy

---

## 📝 Implementation Plan

### Changes Made:

1. **Reward Function (`src/trading_env.py`):**
   - ✅ Reduced risk penalty: 0.5 → 0.2 (further reduced to 0.025 in calculation)
   - ✅ Reduced drawdown penalty: 0.3 → 0.1 (further reduced to 0.015 in calculation)
   - ✅ Increased drawdown threshold: 15% → 20%
   - ✅ Added exploration bonus: +0.0001 per position size
   - ✅ Added small penalty for not trading: -0.00005 when flat
   - ✅ Reduced holding cost: 0.001 → 0.0005
   - ✅ Increased profit bonus: 0.1 → 0.2
   - ✅ Added loss mitigation: Reduce loss penalty by 30%
   - ✅ Reduced reward scaling: 10x → 5x

2. **Training Config (`configs/train_config.yaml`):**
   - ✅ Increased learning rate: 0.0003 → 0.0005
   - ✅ Increased entropy coefficient: 0.01 → 0.025
   - ✅ Updated reward config: risk_penalty 0.5 → 0.2, drawdown_penalty 0.3 → 0.1

### Expected Impact:

**Before (at 480K timesteps):**
- Mean Reward: -40.65
- Best Reward: -19.48
- Agent behavior: Likely avoiding trading

**After (with optimizations):**
- Mean Reward: Should become positive within next 50K-100K timesteps
- Best Reward: Should become positive
- Agent behavior: Should start trading more actively
- Episodes: Should complete more frequently

### Next Steps:

1. **Resume training** with new config
2. **Monitor** for next 50K-100K timesteps
3. **Expected timeline:**
   - 50K steps (530K total): First positive rewards should appear
   - 100K steps (580K total): Mean reward should become positive
   - 150K steps (630K total): Consistent positive rewards

### If Still Not Positive After 100K More Steps:

Additional optimizations to consider:
- Further reduce penalties
- Increase exploration bonus
- Add reward shaping (bonus for closing profitable trades)
- Consider curriculum learning (start with easier market conditions)

