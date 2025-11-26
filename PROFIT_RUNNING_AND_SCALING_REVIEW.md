# Review: Let Profits Run & Scale-In with Multiple Confluence

## Summary

Both features exist in the codebase but may not be fully active during training. Here's the current state:

---

## 1. Let Profits Run (Trailing Stops) ✅ **IMPLEMENTED** but ❌ **NOT IN TRAINING**

### Location
- **File**: `src/risk_manager.py`
- **Method**: `_apply_break_even_logic()` (lines 345-400)

### How It Works
```python
# Break-even activation at 0.6% profit
activation_pct = 0.006  # 0.6%

# Trailing stop at 0.15% below peak
trail_pct = 0.0015  # 0.15%

# Logic:
1. When profit >= 0.6%: Activate break-even
2. Track max_favorable price (highest/lowest price reached)
3. Set trail_price = max_favorable * (1 - trail_pct) for longs
4. If price drops below trail_price: Exit position (let profits run)
5. Scale out if confluence drops below threshold
```

### Configuration (`configs/train_config_adaptive.yaml`)
```yaml
risk_management:
  break_even:
    enabled: true
    activation_pct: 0.006      # Activate at 0.6% profit
    trail_pct: 0.0015          # Trail 0.15% below peak
    scale_out_fraction: 0.5    # Scale out 50% if confluence drops
    scale_out_min_confluence: 2 # Minimum confluence to maintain full position
    free_trade_fraction: 0.5   # Let 50% run as "free trade"
```

### Current Status
- ✅ **Implemented**: Logic exists in `risk_manager.py`
- ❌ **Not Used in Training**: `risk_manager.py` is only used in **live trading** (`live_trading.py`)
- ❌ **Missing in Training**: `trading_env.py` does NOT have trailing stop or break-even logic
- ✅ **Used in Live Trading**: Active when trading via NT8

### What's Missing in Training Environment
The `trading_env.py` only has:
- Stop-loss enforcement (caps losses)
- No break-even logic
- No trailing stops
- No "let profits run" mechanism

---

## 2. Scale-In with Multiple Confluence ✅ **IMPLEMENTED** but ⚠️ **CONDITIONAL**

### Location
- **File**: `src/decision_gate.py`
- **Method**: `_apply_position_sizing()` (lines 304-396)
- **Method**: `_lookup_scale_multiplier()` (lines 295-302)

### How It Works
```python
# Scale multipliers based on confluence count
scale_multipliers = {
    1: 1.0,    # 1 confluence = baseline size
    2: 1.15,   # 2 confluences = 15% larger
    3: 1.3     # 3+ confluences = 30% larger
}

# Logic:
1. Count confluence signals (RL + Swarm + Elliott + Timeframes + etc.)
2. Look up scale multiplier based on count
3. Apply multiplier to RL action: scaled_action = action * multiplier
4. Additional adjustments based on confidence, win rate, regime
```

### Configuration (`configs/train_config_adaptive.yaml`)
```yaml
decision_gate:
  position_sizing:
    enabled: true
    scale_multipliers:
      '1': 1.0    # 1 confluence = baseline
      '2': 1.15   # 2 confluences = +15%
      '3': 1.3    # 3+ confluences = +30%
    max_scale: 1.3
    min_scale: 0.3
```

### Current Status
- ✅ **Implemented**: Logic exists in `decision_gate.py`
- ⚠️ **Conditional**: Only works if:
  1. `training.use_decision_gate: true` is set (✅ Currently enabled)
  2. Swarm is enabled OR confluence is calculated (⚠️ Swarm is disabled in training)
  3. Position sizing is enabled (✅ Currently enabled)

### Issue: Position Sizing May Not Work in Training
According to the code review:
- DecisionGate is enabled in training (`use_decision_gate: true`)
- But swarm is **disabled** during training
- Position sizing uses confluence count, which may be 0 without swarm
- Scale multipliers require `confluence_count > 0` to apply

---

## 3. Recommendations

### For "Let Profits Run" (Trailing Stops)

**Option 1: Add to Training Environment** (RECOMMENDED)
- Add break-even and trailing stop logic to `trading_env.py`
- Similar to stop-loss enforcement, but for profits
- Allows agent to learn to let profits run during training

**Option 2: Keep in Live Trading Only**
- Current approach: Only used in live trading
- Training learns without trailing stops
- Risk: Training may not learn to hold winners long enough

### For "Scale-In with Multiple Confluence"

**Option 1: Enable Simplified Confluence in Training**
- Calculate confluence from available signals (even without swarm)
- Use RL confidence, quality score, or other metrics as confluence
- Apply position sizing based on simplified confluence count

**Option 2: Keep Current Behavior**
- Position sizing works when swarm is enabled
- During training (swarm disabled), no scaling applied
- Only baseline position sizes used

---

## 4. Current Implementation Details

### Trailing Stops (`risk_manager.py`)

**Activation**:
```python
# Activates when profit >= 0.6%
if move_pct >= activation_pct:  # 0.006 = 0.6%
    state["break_even_active"] = True
```

**Trailing Logic**:
```python
# For longs: Trail 0.15% below highest price
state["max_favorable"] = max(state["max_favorable"], market_price)
state["trail_price"] = state["max_favorable"] * (1 - trail_pct)  # 0.15% below peak

# Exit if price drops below trail
if market_price <= state["trail_price"]:
    return 0.0  # Close position
```

**Scale-Out on Confluence Drop**:
```python
# If confluence drops below 2, scale out to 50% position
if confluence_count <= min_confluence:  # min_confluence = 2
    target_size = protected_size * free_fraction  # 50% of protected size
```

### Position Sizing (`decision_gate.py`)

**Confluence Counting**:
```python
confluence_count = sum([
    rl_signal,           # RL recommendation
    swarm_signal,        # Swarm recommendation
    elliott_signal,      # Elliott Wave
    timeframe_alignment, # Multiple timeframes agree
    # ... other signals
])
```

**Scale Multiplier Lookup**:
```python
if confluence_count == 1:
    scale = 1.0   # Baseline
elif confluence_count == 2:
    scale = 1.15  # +15%
elif confluence_count >= 3:
    scale = 1.3   # +30%
```

**Applied Scaling**:
```python
scaled_action = original_action * scale_factor * confidence_factor * win_rate_factor * regime_factor
```

---

## 5. Action Items

### Immediate Actions

1. **Verify Position Sizing in Training**
   - Check if confluence is calculated without swarm
   - Test if scale multipliers are applied during training
   - Add logging to verify scaling behavior

2. **Add Trailing Stops to Training Environment**
   - Port break-even logic from `risk_manager.py` to `trading_env.py`
   - Enable "let profits run" during training
   - Ensure agent learns to hold winners longer

### Future Enhancements

1. **Simplify Confluence for Training**
   - Calculate confluence from RL confidence + quality metrics
   - Enable position sizing even without swarm

2. **Make Trailing Stops Adaptive**
   - Adjust trail_pct based on volatility
   - Make activation_pct adaptive (e.g., 1R = activation)

---

## 6. Files to Review

### Trailing Stops
- `src/risk_manager.py` - Lines 345-400 (break-even logic)
- `src/trading_env.py` - Missing trailing stop implementation
- `configs/train_config_adaptive.yaml` - Lines 181-187 (break-even config)

### Position Sizing
- `src/decision_gate.py` - Lines 295-396 (position sizing logic)
- `src/decision_gate.py` - Lines 191-293 (confluence calculation)
- `configs/train_config_adaptive.yaml` - Lines 277-290 (position sizing config)
- `src/train.py` - Lines 331-352 (DecisionGate initialization)

---

## Conclusion

Both features exist but have limitations:
- **Trailing Stops**: ✅ Implemented but ❌ Not in training environment
- **Position Sizing**: ✅ Implemented but ⚠️ May not work without swarm

**Recommendation**: Add trailing stops to training environment and ensure position sizing works with simplified confluence calculation.

