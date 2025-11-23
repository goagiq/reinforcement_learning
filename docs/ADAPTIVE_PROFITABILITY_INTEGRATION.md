# Adaptive Profitability Integration Plan

## Analysis: What Should Be Adaptive?

### ✅ Should Be Adaptive

1. **Risk/Reward Ratio Threshold** (`min_risk_reward_ratio`)
   - **Why**: If system is losing money, we need stricter R:R (e.g., 1.5 → 2.0)
   - **Why**: If system is profitable, we can relax slightly (e.g., 1.5 → 1.3) to allow more trades
   - **Adaptive Logic**: 
     - If `avg_win / avg_loss < 1.5`: Increase threshold (tighter)
     - If `avg_win / avg_loss >= 2.0`: Decrease threshold (relaxed)
     - Range: 1.3 to 2.5

2. **Quality Filter Thresholds** (`min_action_confidence`, `min_quality_score`)
   - **Why**: If too many trades, tighten filters to improve quality
   - **Why**: If no trades, relax filters to allow exploration
   - **Adaptive Logic**:
     - If trades/episode > 2.0: Increase thresholds (tighter)
     - If trades/episode < 0.3: Decrease thresholds (relaxed)
     - Range: min_action_confidence (0.1 to 0.2), min_quality_score (0.3 to 0.5)

### ❌ Should NOT Be Adaptive

1. **Stop Loss** (`stop_loss_pct`)
   - **Why**: Hard risk management rule - safety mechanism
   - **Why**: Should never be relaxed (would increase risk)
   - **Fixed**: 2% stop loss (non-negotiable)

## Implementation Plan

### 1. Add Adaptive R:R Threshold Adjustment

**Location**: `src/adaptive_trainer.py` - `_analyze_and_adjust()` method

**Logic**:
```python
# Check risk/reward ratio
if avg_win > 0 and avg_loss > 0:
    current_rr_ratio = avg_win / avg_loss
    
    # If losing money, tighten R:R threshold
    if current_rr_ratio < 1.5:
        # Increase min_risk_reward_ratio (tighter)
        new_rr_threshold = min(2.5, current_rr_threshold + 0.1)
    
    # If very profitable, can relax slightly
    elif current_rr_ratio >= 2.0:
        # Decrease min_risk_reward_ratio (relaxed)
        new_rr_threshold = max(1.3, current_rr_threshold - 0.05)
```

### 2. Add Adaptive Quality Filter Adjustment

**Location**: `src/adaptive_trainer.py` - `_analyze_and_adjust()` method

**Logic**:
```python
# Check trade count
trades_per_episode = snapshot.total_trades / max(1, eval_episodes)

# If too many trades, tighten filters
if trades_per_episode > 2.0:
    # Increase min_action_confidence and min_quality_score
    new_min_confidence = min(0.2, current_min_confidence + 0.01)
    new_min_quality = min(0.5, current_min_quality + 0.02)

# If no trades, relax filters
elif trades_per_episode < 0.3:
    # Decrease min_action_confidence and min_quality_score
    new_min_confidence = max(0.1, current_min_confidence - 0.01)
    new_min_quality = max(0.3, current_min_quality - 0.02)
```

### 3. Store Adaptive Parameters

**Location**: `src/adaptive_trainer.py` - Add to `AdaptiveConfig` and `AdaptiveTrainer`

**New Fields**:
- `current_min_risk_reward_ratio: float = 1.5`
- `current_min_action_confidence: float = 0.15`
- `current_min_quality_score: float = 0.4`

### 4. Apply Adaptive Parameters to Environment

**Location**: `src/trading_env.py` - Read from adaptive config file

**Method**: Store adaptive parameters in `logs/adaptive_training/current_reward_config.json` and read in environment

---

## Benefits

1. **Automatic Optimization**: System adjusts itself based on performance
2. **Better Trade Quality**: Tightens filters when too many trades
3. **Better Exploration**: Relaxes filters when no trades
4. **Profitability Focus**: Adjusts R:R threshold based on actual performance
5. **No Manual Intervention**: All adjustments happen automatically

---

## Configuration

Add to `AdaptiveConfig`:
```python
# Risk/reward ratio adjustment
rr_adjustment_enabled: bool = True
min_rr_threshold: float = 1.3  # Minimum R:R threshold
max_rr_threshold: float = 2.5  # Maximum R:R threshold
rr_adjustment_rate: float = 0.1  # How much to adjust

# Quality filter adjustment
quality_filter_adjustment_enabled: bool = True
min_action_confidence_range: Tuple[float, float] = (0.1, 0.2)
min_quality_score_range: Tuple[float, float] = (0.3, 0.5)
quality_adjustment_rate: float = 0.01  # How much to adjust
```

