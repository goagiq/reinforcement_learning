# Retraining Configuration Changes - Anti-Saturation Fixes

## Changes Applied to `configs/train_config_adaptive.yaml`

### 1. Increased Exploration (Entropy Coefficient)
**Changed:**
```yaml
entropy_coef: 0.15  # INCREASED from 0.05 (3x increase)
```

**Why:** The model was saturating to always output maximum actions (±1.0). Higher entropy encourages exploration and prevents premature convergence to extreme actions.

**Impact:** Model will explore more action space, preventing saturation.

---

### 2. Added Action Diversity Rewards
**Added:**
```yaml
action_diversity_bonus: 0.01  # Reward diverse actions
constant_action_penalty: 0.05  # Penalize always same action
```

**Why:** Directly incentivizes the model to produce diverse actions rather than always the same value.

**Impact:** Model receives positive reward for varying actions and negative reward for constant actions.

---

### 3. Increased Exploration Bonus
**Changed:**
```yaml
exploration_bonus_scale: 5.0e-05  # Increased from 1e-5 (5x increase)
```

**Why:** Further encourages exploration during early training.

**Impact:** More exploration in early stages, helping model learn diverse strategies.

---

### 4. Disabled Transfer Learning
**Changed:**
```yaml
transfer_learning: false  # Starting from scratch
transfer_checkpoint: null
```

**Why:** All existing checkpoints are saturated. Starting fresh with new hyperparameters.

**Impact:** Clean slate - no bad patterns from previous training.

---

## Expected Results

### Before (Saturated Model):
- ❌ Always outputs action = ±1.0
- ❌ No action diversity
- ❌ 0% win rate
- ❌ All trades lose

### After (With Fixes):
- ✅ Diverse actions (varying between -1.0 and +1.0)
- ✅ Action variance > 0
- ✅ Model explores different strategies
- ✅ Better learning and adaptation

---

## Monitoring During Training

Watch for these metrics to ensure saturation is prevented:

1. **Action Variance**: Should be > 0.1 (not always same value)
2. **Action Distribution**: Should show values across [-1.0, 1.0] range
3. **Entropy**: Should remain relatively high (not decreasing to near zero)
4. **Reward**: Should show learning progress (not stuck)

### Early Warning Signs of Saturation:
- ⚠️ Action variance < 0.01
- ⚠️ All actions near ±1.0
- ⚠️ Entropy decreasing rapidly to near zero
- ⚠️ Reward not improving

If saturation starts again:
- Increase `entropy_coef` further (0.2-0.3)
- Increase `action_diversity_bonus` (0.02-0.05)
- Increase `constant_action_penalty` (0.1-0.2)

---

## Training Command

The config is ready for retraining. Use your UI or:

```bash
python src/train.py \
  --config configs/train_config_adaptive.yaml \
  --device cuda
```

**Note:** No checkpoint needed - starting from scratch.

---

## Summary of Changes

| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| `entropy_coef` | 0.05 | 0.15 | +200% |
| `exploration_bonus_scale` | 1e-5 | 5e-5 | +400% |
| `action_diversity_bonus` | (none) | 0.01 | NEW |
| `constant_action_penalty` | (none) | 0.05 | NEW |
| `transfer_learning` | true | false | DISABLED |

**Status:** ✅ **Ready for retraining from scratch**

