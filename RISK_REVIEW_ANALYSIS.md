# Risk Review Analysis: Current vs GitHub Repo

Based on the [GitHub repository](https://github.com/goagiq/reinforcement_learning) structure and current codebase analysis.

## 🔍 Current Risk Management Settings

### Risk Limits (from `configs/train_config_adaptive.yaml`)

```yaml
risk_management:
  max_position_size: 1.0              # Maximum position size (normalized)
  max_drawdown: 0.2                   # Maximum drawdown (20%)
  stop_loss_atr_multiplier: 2.0        # Stop loss as multiple of ATR
  initial_capital: 100000.0           # Starting capital
  commission: 2.0                     # Commission per contract per side
  max_position_fraction_of_balance: 0.02  # Max risk per trade (2% of equity)
  position_value_per_unit: 5000.0     # Notional value per unit
```

### Reward Function Risk Controls

```yaml
environment.reward:
  stop_loss_pct: 0.025                # 2.5% stop loss
  min_risk_reward_ratio: 2.0          # Minimum 2.0:1 R:R (recently increased)
  transaction_cost: 0.0001            # 0.01% transaction cost
  risk_penalty: 0.05                   # Risk penalty
  drawdown_penalty: 0.07               # Drawdown penalty
  max_consecutive_losses: 10           # Stop after 10 consecutive losses
```

### Break-Even & Position Management

```yaml
risk_management.break_even:
  enabled: true
  activation_pct: 0.006                # Move to break-even after 0.6% profit
  trail_pct: 0.0015                    # Trail stop 0.15% behind price
  scale_out_fraction: 0.5              # Reduce position by 50% when confluences drop
  scale_out_min_confluence: 2          # Trigger scale-out when confluences <= 2
  free_trade_fraction: 0.5             # Maintain 50% size once trade is free
```

## ⚠️ Risk Issues Identified

### 1. **Commission Cost Mismatch**

**Problem:**
- `risk_management.commission: 2.0` (per contract per side)
- `environment.reward.transaction_cost: 0.0001` (0.01%)
- `transaction_costs.commission_rate: 0.0003` (0.03%)

**Analysis:**
- Multiple commission settings that may conflict
- Actual commission in trading journal: **$28.51 per trade** (31% of net win)
- This is causing rapid losses as identified in previous analysis

**Recommendation:**
- Standardize commission calculation
- Use single source of truth for commission costs
- Ensure R:R accounts for actual commission (already fixed: 2.0:1)

### 2. **Stop Loss Settings**

**Current:**
- `stop_loss_pct: 0.025` (2.5%)
- `stop_loss_atr_multiplier: 2.0`

**GitHub Repo (from comparison):**
- `stop_loss_pct: 0.02` (2.0%)

**Analysis:**
- Current 2.5% is reasonable (was 1.5% - too tight, 4.0% - too loose)
- ATR multiplier of 2.0 is standard

**Status:** ✅ Acceptable

### 3. **Position Sizing Risk**

**Current:**
- `max_position_fraction_of_balance: 0.02` (2% of equity per trade)
- `max_position_size: 1.0` (normalized)

**Analysis:**
- 2% risk per trade is conservative and good
- Max position size of 1.0 is reasonable

**Status:** ✅ Acceptable

### 4. **Drawdown Protection**

**Current:**
- `max_drawdown: 0.2` (20% maximum drawdown)
- `drawdown_penalty: 0.07` (7% penalty in reward function)

**GitHub Repo:**
- `max_drawdown: 0.2` (20%)

**Analysis:**
- 20% max drawdown is standard
- Drawdown penalty encourages risk management

**Status:** ✅ Acceptable

### 5. **Risk/Reward Ratio**

**Current:**
- `min_risk_reward_ratio: 2.0` (recently increased from 1.5)
- Enforcement floor: `1.0` (rejects trades below break-even)

**GitHub Repo:**
- `min_risk_reward_ratio: 1.5` or `None`

**Analysis:**
- **2.0:1 is CORRECT** - accounts for commission costs (31% of net win)
- Current actual R:R: **0.71:1** (way too low)
- Need to improve actual R:R toward 2.0:1

**Status:** ⚠️ Setting is correct, but actual performance is poor

### 6. **Daily Loss Limits**

**Current:**
- `max_daily_loss: 0.05` (5% daily loss limit) - **MISSING from config!**
- Default in `RiskManager`: `0.05` (5%)

**Analysis:**
- 5% daily loss limit is good risk management
- Should be explicitly set in config

**Recommendation:**
- Add `max_daily_loss: 0.05` to `risk_management` section

### 7. **Consecutive Loss Protection**

**Current:**
- `max_consecutive_losses: 10` (stop after 10 consecutive losses)

**GitHub Repo:**
- `max_consecutive_losses: 3` or `None`

**Analysis:**
- 10 consecutive losses is very lenient
- May allow too much loss before stopping

**Recommendation:**
- Consider reducing to 5-7 consecutive losses

## 📊 Risk Management Comparison

| Risk Parameter | Current | GitHub Repo | Status |
|----------------|---------|-------------|--------|
| **Max Drawdown** | 20% | 20% | ✅ Same |
| **Stop Loss %** | 2.5% | 2.0% | ✅ Similar |
| **Position Risk** | 2% | 2% | ✅ Same |
| **R:R Requirement** | 2.0:1 | 1.5:1 | ⚠️ Higher (correct for commission) |
| **Daily Loss Limit** | 5% (default) | Not specified | ⚠️ Should be explicit |
| **Consecutive Losses** | 10 | 3 | ⚠️ Too lenient |
| **Commission** | Multiple settings | Simple | ❌ Needs standardization |

## 🎯 Recommendations

### Priority 1: Fix Commission Standardization

```yaml
# Standardize commission - use ONE setting
risk_management:
  commission: 2.0  # Per contract per side (for risk calculations)
  
environment.reward:
  transaction_cost: 0.0001  # 0.01% (for reward function)
  
transaction_costs:
  commission_rate: 0.0003  # 0.03% (for actual trading)
  # NOTE: This is higher than transaction_cost - may need alignment
```

**Action:** Review and align all commission settings to use consistent values.

### Priority 2: Add Missing Risk Limits

```yaml
risk_management:
  max_drawdown: 0.2
  max_daily_loss: 0.05  # ADD THIS - currently using default
  max_position_size: 1.0
  max_position_fraction_of_balance: 0.02
```

### Priority 3: Tighten Consecutive Loss Protection

```yaml
environment.reward:
  max_consecutive_losses: 5  # Reduce from 10 to 5
```

### Priority 4: Verify Break-Even Logic

Current break-even settings:
- `activation_pct: 0.006` (0.6%) - Move to break-even after 0.6% profit
- `trail_pct: 0.0015` (0.15%) - Trail stop 0.15% behind

**Analysis:**
- These settings are reasonable
- Helps protect profits once trades are in profit

**Status:** ✅ Acceptable

## 🔒 Risk Management Best Practices

### ✅ Good Practices Currently Implemented

1. **Position Sizing:** 2% risk per trade (conservative)
2. **Drawdown Limit:** 20% maximum (standard)
3. **Stop Loss:** 2.5% (reasonable)
4. **R:R Requirement:** 2.0:1 (accounts for commission)
5. **Break-Even Logic:** Protects profits
6. **Daily Loss Limit:** 5% (good protection)

### ⚠️ Areas for Improvement

1. **Commission Standardization:** Multiple conflicting settings
2. **Consecutive Loss Limit:** Too lenient (10 → 5)
3. **Daily Loss Limit:** Should be explicit in config
4. **Actual R:R Performance:** 0.71:1 vs required 2.0:1 (needs improvement)

## 📈 Expected Impact of Fixes

### With Standardized Commission:
- Clearer cost calculation
- Better R:R enforcement
- More accurate profitability analysis

### With Tighter Consecutive Loss Limit:
- Faster stop after losing streaks
- Reduced risk of large drawdowns
- Better risk management

### With Explicit Daily Loss Limit:
- Clear risk boundaries
- Better monitoring
- Easier configuration review

## 🎯 Action Items

1. ✅ **DONE:** Increased R:R to 2.0:1 (accounts for commission)
2. ✅ **DONE:** Tightened R:R enforcement floor to 1.0
3. ⚠️ **TODO:** Standardize commission settings
4. ⚠️ **TODO:** Add explicit `max_daily_loss` to config
5. ⚠️ **TODO:** Reduce `max_consecutive_losses` from 10 to 5
6. ⚠️ **TODO:** Monitor actual R:R improvement toward 2.0:1

## 📝 Summary

**Overall Risk Management Status:** ⚠️ **Good foundation, needs refinement**

**Key Strengths:**
- Conservative position sizing (2%)
- Reasonable drawdown limits (20%)
- Good stop loss settings (2.5%)
- Correct R:R requirement (2.0:1) for commission costs

**Key Weaknesses:**
- Commission settings not standardized
- Actual R:R performance poor (0.71:1 vs 2.0:1 required)
- Consecutive loss limit too lenient
- Daily loss limit not explicit in config

**Most Critical Issue:**
- **Actual R:R is 0.71:1** - way below required 2.0:1
- This is causing rapid losses despite good risk management settings
- Need to improve trade management to achieve better R:R

