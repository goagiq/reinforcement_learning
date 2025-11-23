# Investigation Findings - Training Issues

**Date**: After Windows Update Reboot  
**Status**: 🔍 Investigation In Progress

---

## ✅ CONFIRMED FINDINGS

### 1. Adaptive Training Status
- **Status**: ✅ **ENABLED**
- **Configuration**: `adaptive_training.enabled: true`
- **Impact on Episodes**: ✅ **DOES NOT cause short episodes**
  - Trading pause auto-resumes after 100 steps
  - Episodes continue normally even when trading is paused
  - Only prevents new trades, doesn't terminate episodes

**Conclusion**: Adaptive training is working as designed and is NOT the cause of 20-step episodes.

### 2. Capital Preservation Logic
- **Status**: ✅ **WORKING CORRECTLY**
- **Mechanism**: Trading pauses after 10 consecutive losses
- **Auto-Resume**: After 100 steps (for training)
- **Impact**: Does NOT terminate episodes early

**Conclusion**: Capital preservation logic is NOT causing short episodes.

### 3. Max Consecutive Losses Fix
- **Status**: ✅ **FIX IS APPLIED**
- **Location**: `src/trading_env.py` line 598
- **Definition**: Before first use (line 628)
- **Fix**: `max_consecutive_losses` is defined before use in stop loss logic

**Conclusion**: The documented fix is correctly applied in the code.

---

## 🔍 HYPOTHESIS: Root Cause of 20-Step Episodes

### Most Likely Cause: Exception in State Feature Extraction

**Evidence**:
1. Episodes terminating at exactly 20 steps
2. `lookback_bars = 20` (coincidence?)
3. Code has exception handling that terminates on IndexError/KeyError (line 828)
4. Exception likely occurs when accessing data at step 20

**Code Location**: `src/trading_env.py` lines 821-829
```python
try:
    next_state = self._get_state_features(safe_step)
except (IndexError, KeyError) as e:
    # CRITICAL FIX: Catch exceptions in state feature extraction and terminate gracefully
    import sys
    print(f"[ERROR] Exception in _get_state_features at step {self.current_step}: {e}", flush=True)
    sys.stdout.flush()
    terminated = True
    next_state = np.zeros(self.state_dim, dtype=np.float32)
```

**Why Step 20?**
- Episode starts at `current_step = lookback_bars` (20)
- First few steps may work, then exception at step 20
- Or exception occurs when trying to access lookback data
- Could be data boundary issue or missing key in data

---

## 📋 INVESTIGATION STATUS

### ✅ Completed
- [x] Verified adaptive training is enabled
- [x] Confirmed adaptive training doesn't cause short episodes
- [x] Verified max_consecutive_losses fix is applied
- [x] Reviewed capital preservation logic

### 🔄 In Progress
- [ ] Check backend logs for exception messages
- [ ] Test episode termination in isolation
- [ ] Verify data boundaries

### ⏳ Pending
- [ ] Review quality filter rejection reasons
- [ ] Calculate average win vs loss sizes
- [ ] Verify stop loss enforcement
- [ ] Test with relaxed filters

---

## 🎯 NEXT STEPS (Priority Order)

### 1. Check Backend Logs (URGENT)
**Action**: Look for exception messages in training logs
```bash
# Check for exception messages
grep -i "exception\|error\|warning" logs/*.log | tail -50
```

**Look for**:
- `[ERROR] Exception in _get_state_features`
- `[WARNING] Episode terminating early`
- `IndexError`
- `KeyError`
- `UnboundLocalError`

### 2. Test Episode Termination (URGENT)
**Action**: Create test script to reproduce 20-step issue
```python
# test_episode_termination.py
from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor
import yaml

# Load config and create environment
with open('configs/train_config_adaptive.yaml') as f:
    config = yaml.safe_load(f)

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
        if 'error' in step_info:
            print(f"Error: {step_info['error']}")
        break
```

### 3. Verify Data Boundaries (HIGH PRIORITY)
**Action**: Check if data is long enough and if episode start position is valid
```python
# Check data length vs requirements
from src.data_extraction import DataExtractor
import yaml

with open('configs/train_config_adaptive.yaml') as f:
    config = yaml.safe_load(f)

extractor = DataExtractor(config)
data = extractor.load_data()

primary_data = data[min(config['environment']['timeframes'])]
max_steps = config['environment']['max_episode_steps']
lookback = config['environment']['lookback_bars']
required = max_steps + lookback

print(f"Data length: {len(primary_data)}")
print(f"Required: {required}")
print(f"Margin: {len(primary_data) - required}")
```

### 4. Review Quality Filters (HIGH PRIORITY)
**Action**: Add logging to see why trades are rejected
- Check `src/decision_gate.py` for rejection reasons
- Check `src/quality_scorer.py` for quality score calculations
- Add metrics to track:
  - Total trade attempts
  - Rejected by quality filters
  - Rejected by DecisionGate
  - Rejected by action threshold
  - Rejected by trading_paused

### 5. Calculate Win/Loss Sizes (MEDIUM PRIORITY)
**Action**: Add metrics to track average win vs loss
- Average win size
- Average loss size
- Win/loss ratio
- Risk/reward ratio

### 6. Verify Stop Loss (MEDIUM PRIORITY)
**Action**: Check if stop loss is actually enforced
- Review stop loss logic in `src/trading_env.py` (lines 606-642)
- Verify stop loss percentage is calculated correctly
- Check if positions are actually closed when stop loss is hit

---

## 📊 CURRENT METRICS SUMMARY

### Episode Length
- **Latest**: 20 steps ❌ (0.2% of expected)
- **Mean**: 9,980 steps ✅ (normal)
- **Issue**: Episodes terminating very early

### Trade Count
- **Total**: 10 trades in 380 episodes
- **Rate**: 0.026 trades/episode ❌
- **Target**: 0.5-1.0 trades/episode
- **Gap**: Missing ~180-370 trades

### Profitability
- **Mean PnL (Last 10)**: -$2,015.06 ❌
- **Win Rate**: 44.4% ✅ (above breakeven)
- **Issue**: Average loss size >> Average win size

### Rewards
- **Latest**: -0.0038 ❌
- **Mean (Last 10)**: -1.70 ❌
- **Issue**: Rewards still negative

---

## 🔧 RECOMMENDED FIXES (After Investigation)

### Fix 1: Exception Handling (If Exception Found)
- Add more detailed error logging
- Fix data boundary checks
- Ensure all data keys exist before access

### Fix 2: Quality Filters (If Too Strict)
- Temporarily relax filters to test
- Add metrics to track rejection reasons
- Optimize based on rejection patterns

### Fix 3: Stop Loss (If Not Working)
- Verify stop loss enforcement
- Check position closing logic
- Ensure stop loss percentage is correct

### Fix 4: Reward Function (If Too Punitive)
- Review reward components
- Balance penalties vs rewards
- Adjust inaction penalty if needed

---

## 📝 NOTES

### Adaptive Training
- ✅ Enabled and working correctly
- ✅ Does NOT cause short episodes
- ✅ Auto-adjusts parameters based on performance
- ✅ May help with trade count if filters are too strict

### Capital Preservation
- ✅ Trading pause mechanism working
- ✅ Auto-resumes after 100 steps
- ✅ Does NOT terminate episodes
- ⚠️ May reduce trade count if pausing frequently

### Episode Termination
- 🔍 Most likely cause: Exception in state feature extraction
- 🔍 Need to check logs and test in isolation
- 🔍 Could be data boundary issue

---

**Status**: 🔍 **INVESTIGATION IN PROGRESS**  
**Next Update**: After checking logs and testing episode termination

