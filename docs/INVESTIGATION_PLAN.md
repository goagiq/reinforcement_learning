# Investigation Plan - Training Issues

**Date**: After Windows Update Reboot  
**Status**: 🔴 Critical Issues Identified

---

## Summary

**Adaptive Training**: ✅ **ENABLED** - Adaptive training system is running  
**Capital Preservation**: ⚠️ **PARTIAL** - Trading can pause after consecutive losses, but auto-resumes after 100 steps

**Key Finding**: Adaptive training is enabled, but the short episodes (20 steps) are NOT caused by trading_paused logic. Episodes continue even when trading is paused - they just don't allow new trades.

---

## Issue 1: Episodes Terminating at 20 Steps (CRITICAL)

### Current State
- Latest Episode: **20 steps** (0.2% of expected 10,000)
- Mean Episode Length: 9,980 steps (normal)
- Pattern: Episodes terminating very early

### Possible Causes (from code analysis)

1. **Exception in `_get_state_features()`** (Most Likely)
   - Code has try/except that catches IndexError/KeyError
   - If exception occurs, episode terminates early (line 828)
   - Could be happening at step 20 due to data boundary issue

2. **Data Boundary Issue**
   - Code checks: `safe_step >= len(primary_data) - lookback_bars`
   - If data is shorter than expected, episodes terminate early
   - Need to verify data length vs episode start position

3. **UnboundLocalError (Supposedly Fixed)**
   - Docs say `max_consecutive_losses` was moved before first use
   - But episodes still terminating at 20 steps
   - Need to verify fix is actually in code

### Investigation Steps

#### Step 1.1: Check Backend Logs
```bash
# Check for exception messages
grep -i "exception\|error\|warning" logs/*.log | tail -50
# Or check console output if training is running
```

**Look for**:
- `[ERROR] Exception in _get_state_features`
- `[WARNING] Episode terminating early`
- `UnboundLocalError`
- `IndexError`
- `KeyError`

#### Step 1.2: Verify Fix is Applied
```python
# Check if max_consecutive_losses is defined before use
grep -n "max_consecutive_losses" src/trading_env.py
```

**Expected**: Should be defined at line ~591 (before first use at line 598)

#### Step 1.3: Test Episode Termination
```bash
python investigate_short_episodes.py
```

**Or create test script**:
```python
# test_episode_termination.py
from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor
import yaml

# Load config
with open('configs/train_config_adaptive.yaml') as f:
    config = yaml.safe_load(f)

# Create environment
extractor = DataExtractor(config)
data = extractor.load_data()
env = TradingEnvironment(data, config)

# Test episode
state, info = env.reset()
for step in range(100):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, step_info = env.step(action)
    
    if terminated:
        print(f"Episode terminated at step {step+1}")
        print(f"Reason: {step_info.get('termination_reason', 'unknown')}")
        break
    
    if step_info.get('error'):
        print(f"Error at step {step+1}: {step_info['error']}")
        break
```

#### Step 1.4: Check Data Length
```python
# Check if data is long enough
from src.data_extraction import DataExtractor
import yaml

with open('configs/train_config_adaptive.yaml') as f:
    config = yaml.safe_load(f)

extractor = DataExtractor(config)
data = extractor.load_data()

primary_data = data[min(config['environment']['timeframes'])]
print(f"Data length: {len(primary_data)}")
print(f"Max episode steps: {config['environment']['max_episode_steps']}")
print(f"Lookback bars: {config['environment']['lookback_bars']}")
print(f"Required length: {config['environment']['max_episode_steps'] + config['environment']['lookback_bars']}")
```

---

## Issue 2: Extremely Low Trade Count (CRITICAL)

### Current State
- **10 trades in 380 episodes** = 0.026 trades/episode
- **Target**: 0.5-1.0 trades/episode (should have 190-380 trades)
- **Gap**: Missing ~180-370 trades

### Possible Causes

1. **Quality Filters Too Strict**
   - `min_action_confidence: 0.08` (reduced from 0.12)
   - `min_quality_score: 0.25` (reduced from 0.35)
   - Still may be too strict

2. **DecisionGate Rejecting Too Many Trades**
   - `min_combined_confidence: 0.7` (may be too high)
   - `min_confluence_required: 2` (may be too strict)
   - Quality scorer may be rejecting valid trades

3. **Action Threshold Too High**
   - `action_threshold: 0.01` (1%) - reduced from 0.02
   - May still be too high for continuous action space

4. **Consecutive Loss Limit Pausing Too Often**
   - `max_consecutive_losses: 10` (increased from 5)
   - But if losses happen early, trading pauses for 100 steps
   - With only 20-step episodes, this could be blocking all trades

### Investigation Steps

#### Step 2.1: Review Quality Filter Rejections
```python
# Add logging to see why trades are rejected
# Check src/decision_gate.py and src/quality_scorer.py
# Look for rejection reasons in logs
```

#### Step 2.2: Check DecisionGate Statistics
```python
# Add metrics to track:
# - Total trade attempts
# - Rejected by quality filters
# - Rejected by DecisionGate
# - Rejected by action threshold
# - Rejected by trading_paused
```

#### Step 2.3: Test with Relaxed Filters
```yaml
# Temporarily relax filters to test
quality_filters:
  min_action_confidence: 0.05  # Reduced from 0.08
  min_quality_score: 0.15      # Reduced from 0.25

decision_gate:
  min_combined_confidence: 0.5  # Reduced from 0.7
  min_confluence_required: 1    # Reduced from 2
```

---

## Issue 3: Severe Financial Losses (CRITICAL)

### Current State
- **Mean PnL (Last 10)**: -$2,015.06
- **Current PnL**: -$104.46
- **Mean Equity**: $97,984.94 (down $2,015 from $100k)

### Analysis
- **Win Rate**: 44.4% (above breakeven ~34%) ✅
- **But Mean PnL**: -$2,015 ❌
- **Conclusion**: Average loss size >> Average win size

### Possible Causes

1. **Stop Loss Not Working**
   - `stop_loss_pct: 0.02` (2%) configured
   - But losses may be exceeding 2%
   - Need to verify stop loss is enforced

2. **Position Sizing Issues**
   - Losses may be sized larger than wins
   - Risk/reward ratio may be poor
   - Need to check average win vs loss amounts

3. **Commission Costs**
   - With only 10 trades, commissions shouldn't be $2,015
   - But need to verify commission calculation

### Investigation Steps

#### Step 3.1: Calculate Average Win vs Loss
```python
# Add to metrics tracking:
# - Average win size
# - Average loss size
# - Win/loss ratio
# - Risk/reward ratio
```

#### Step 3.2: Verify Stop Loss Enforcement
```python
# Check src/trading_env.py stop loss logic (lines 606-642)
# Verify stop loss is actually closing positions
# Check if stop loss percentage is calculated correctly
```

#### Step 3.3: Review Position Sizing
```python
# Check if position sizing is consistent
# Verify risk/reward ratio calculations
# Check if losses are being sized larger than wins
```

---

## Issue 4: Negative Rewards (HIGH PRIORITY)

### Current State
- Latest Reward: -0.0038 ❌
- Mean Reward (Last 10): -1.70 ❌

### Possible Causes

1. **Reward Function Too Punitive**
   - Inaction penalty, loss mitigation, drawdown penalty
   - May be penalizing more than rewarding

2. **Poor Trade Quality**
   - Trades that do occur may be low quality
   - Losses outweigh wins in reward calculation

### Investigation Steps

#### Step 4.1: Review Reward Function
```python
# Check src/trading_env.py reward calculation
# Verify reward components:
# - PnL weight
# - Transaction cost
# - Risk penalty
# - Drawdown penalty
# - Exploration bonus
# - Inaction penalty
```

---

## Adaptive Training Status

### ✅ Adaptive Training is ENABLED

**Configuration**:
```yaml
adaptive_training:
  enabled: true
  eval_frequency: 10000
  eval_episodes: 3
  min_trades_per_episode: 0.5
  min_win_rate: 0.35
  target_sharpe: 0.5
  auto_save_on_improvement: true
  improvement_threshold: 0.05
```

**What It Does**:
- Monitors trading activity in real-time
- Adjusts `entropy_coef` if no trades detected
- Adjusts `inaction_penalty` if model stays flat
- Evaluates every 10,000 timesteps
- Auto-saves best model on improvement

**Impact on Episodes**:
- ✅ Does NOT terminate episodes early
- ✅ Trading pause auto-resumes after 100 steps
- ✅ Episodes continue even when trading is paused

**Conclusion**: Adaptive training is NOT causing short episodes.

---

## Capital Preservation Logic

### Trading Pause Mechanism

**How It Works**:
1. After `max_consecutive_losses` (10) consecutive losses, trading pauses
2. While paused, new trades are rejected (position_change = 0)
3. Episode continues normally, just without trades
4. Auto-resumes after 100 steps (for training)
5. Or resumes on next winning trade

**Impact**:
- ✅ Does NOT terminate episodes
- ✅ Episodes continue for full 10,000 steps (if no exceptions)
- ⚠️ May reduce trade count if pausing frequently

**Conclusion**: Capital preservation logic is NOT causing 20-step episodes.

---

## Root Cause Hypothesis

### Most Likely Cause: Exception in State Feature Extraction

**Evidence**:
1. Episodes terminating at exactly 20 steps (lookback_bars = 20)
2. Code has exception handling that terminates on IndexError/KeyError
3. Exception likely occurs when accessing data at step 20

**Why Step 20?**
- `lookback_bars = 20`
- Episode starts at `current_step = lookback_bars` (20)
- First few steps may work, then exception at step 20
- Or exception occurs when trying to access lookback data

**Next Steps**:
1. Check backend logs for exception messages
2. Test episode termination in isolation
3. Verify data boundaries
4. Check if exception is IndexError or KeyError

---

## Investigation Priority

### 🔴 URGENT (Do First)
1. **Check Backend Logs** - Look for exception messages
2. **Test Episode Termination** - Reproduce 20-step issue
3. **Verify Data Boundaries** - Check if data is long enough

### 🟡 HIGH PRIORITY (Do Next)
4. **Review Quality Filters** - Check why trades are rejected
5. **Calculate Win/Loss Sizes** - Understand why losses are so large
6. **Verify Stop Loss** - Check if stop loss is working

### 🟢 MEDIUM PRIORITY (Do Later)
7. **Review Reward Function** - Check if too punitive
8. **Test with Relaxed Filters** - See if trade count increases
9. **Monitor Adaptive Training** - Verify it's working correctly

---

## Action Items

### Immediate (Today)
- [ ] Check backend logs for exceptions
- [ ] Test episode termination in isolation
- [ ] Verify data length vs episode requirements
- [ ] Review quality filter rejection reasons

### Short-Term (This Week)
- [ ] Fix exception causing 20-step episodes
- [ ] Add metrics for win/loss sizes
- [ ] Verify stop loss enforcement
- [ ] Test with relaxed filters

### Medium-Term (Next Week)
- [ ] Optimize quality filters based on results
- [ ] Review reward function parameters
- [ ] Monitor adaptive training adjustments
- [ ] Track improvement metrics

---

## Success Criteria

### Episode Length
- ✅ Episodes complete full 10,000 steps (or until data ends)
- ✅ No exceptions causing early termination
- ✅ Mean episode length: ~10,000 steps

### Trade Count
- ✅ Trade count: 0.5-1.0 trades/episode
- ✅ Total trades: 190-380 trades (for 380 episodes)
- ✅ Quality filters not too strict

### Profitability
- ✅ Mean PnL: Positive (after commissions)
- ✅ Win rate: 60-65%+ (target)
- ✅ Average win size >= Average loss size

### Rewards
- ✅ Mean reward: Positive
- ✅ Reward function balanced (not too punitive)

---

**Status**: 🔴 **INVESTIGATION IN PROGRESS**  
**Next Update**: After checking backend logs and testing episode termination

