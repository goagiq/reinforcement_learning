# Retraining Recommendation for Model Saturation Issue

## Problem Summary

The current model has **saturated** - it always outputs action = 1.0 (maximum) on every step. This is a degenerate state that prevents proper trading.

## Retraining Options Analysis

### Option 1: Retrain from Scratch ❌ **NOT RECOMMENDED**
- **Pros**: Clean slate, no bad patterns
- **Cons**: 
  - Loses all training time (8M+ timesteps)
  - Takes longest time to reach good performance
  - No guarantee it won't saturate again

**Verdict**: Only use if all checkpoints are saturated

### Option 2: Resume from Earlier Checkpoint ✅ **RECOMMENDED**
- **Pros**: 
  - Preserves training progress
  - Can find a checkpoint before saturation occurred
  - Faster than starting from scratch
  - Can adjust hyperparameters to prevent saturation
- **Cons**: 
  - Need to identify when saturation started
  - May need to test multiple checkpoints

**Verdict**: **BEST OPTION** - Use an early checkpoint (before 1M timesteps)

### Option 3: Transfer Learning ⚠️ **CONDITIONAL**
- **Pros**: 
  - Can change architecture while preserving knowledge
  - Useful if architecture needs to change
- **Cons**: 
  - Current architecture is fine ([256, 256, 128])
  - Transfer learning won't fix saturation if source model is saturated
  - Only useful if changing architecture

**Verdict**: Use only if you want to change architecture

## Recommended Approach: **Resume from Early Checkpoint**

### Step 1: Find a Good Checkpoint

Test early checkpoints to find one that:
- ✅ Has varied actions (not always 1.0)
- ✅ Shows learning progress
- ✅ Has reasonable performance

**Recommended checkpoints to test:**
- `checkpoint_100000.pt` (100k steps - early training)
- `checkpoint_200000.pt` (200k steps)
- `checkpoint_500000.pt` (500k steps)
- `checkpoint_1000000.pt` (1M steps)

### Step 2: Adjust Hyperparameters to Prevent Saturation

When resuming, modify the config to prevent saturation:

```yaml
model:
  entropy_coef: 0.1  # INCREASE from 0.05 - encourages exploration
  learning_rate: 0.0001  # Keep same or slightly lower

environment:
  reward:
    # Add penalty for constant actions
    action_diversity_bonus: 0.01  # Reward action diversity
    constant_action_penalty: 0.1  # Penalize always outputting same action
```

### Step 3: Resume Training

**Via API:**
```json
{
  "device": "cuda",
  "config_path": "configs/train_config_adaptive.yaml",
  "checkpoint_path": "models/checkpoint_100000.pt"
}
```

**Via CLI:**
```bash
python src/train.py \
  --config configs/train_config_adaptive.yaml \
  --checkpoint models/checkpoint_100000.pt \
  --device cuda
```

### Step 4: Monitor for Saturation

Watch for these warning signs:
- ⚠️ Actions becoming less diverse over time
- ⚠️ Model always outputting same action value
- ⚠️ Loss not decreasing
- ⚠️ Reward plateauing

If saturation starts again:
- Increase `entropy_coef` further
- Add action diversity reward
- Consider early stopping and resuming from that point

## Alternative: Transfer Learning (If Architecture Change Needed)

If you want to change architecture while retraining:

### When to Use Transfer Learning:
- ✅ You want to change `hidden_dims` (e.g., [256, 256, 128] → [512, 512, 256])
- ✅ You have a good checkpoint with different architecture
- ✅ You want to preserve some learned patterns

### How to Use:
```yaml
training:
  transfer_learning: true
  transfer_checkpoint: models/checkpoint_100000.pt  # Source checkpoint
  transfer_strategy: copy_and_extend  # or interpolate, zero_pad
```

The system will automatically:
1. Detect architecture mismatch
2. Transfer compatible weights
3. Initialize new dimensions intelligently
4. Continue training

## Specific Recommendation

### **RECOMMENDED: Resume from `checkpoint_100000.pt`**

**Why:**
1. Early enough to avoid saturation
2. Has some learning progress
3. Fast to retrain from this point
4. Can adjust hyperparameters to prevent saturation

**Steps:**
1. Test `checkpoint_100000.pt` with diagnostic script:
   ```bash
   python diagnose_model_behavior.py --model models/checkpoint_100000.pt --episodes 2
   ```
2. If actions are diverse (not all 1.0), proceed
3. Update config with anti-saturation measures
4. Resume training from this checkpoint

### Config Changes for Anti-Saturation:

```yaml
model:
  entropy_coef: 0.1  # Increased from 0.05 - more exploration
  learning_rate: 0.0001  # Keep same

environment:
  reward:
    # Add these to prevent saturation
    exploration_bonus_enabled: true
    exploration_bonus_scale: 0.0001  # Increased from 1e-5
    action_diversity_weight: 0.01  # NEW: Reward diverse actions
```

## Testing Strategy

Before committing to full retraining:

1. **Quick Test**: Evaluate checkpoint with diagnostic script
   ```bash
   python diagnose_model_behavior.py --model models/checkpoint_100000.pt --episodes 2
   ```
   - Check if actions vary (not all 1.0)
   - Check if model shows learning

2. **Short Training Test**: Resume for 50k steps
   ```bash
   python src/train.py \
     --config configs/train_config_adaptive.yaml \
     --checkpoint models/checkpoint_100000.pt \
     --device cuda
   ```
   - Monitor for saturation
   - Check action diversity
   - Verify learning is happening

3. **Full Retraining**: If test passes, continue full training

## Summary

| Option | Recommendation | When to Use |
|--------|---------------|-------------|
| **From Scratch** | ❌ Not recommended | Only if all checkpoints are bad |
| **Early Checkpoint** | ✅ **RECOMMENDED** | Best option - preserves progress, fixes saturation |
| **Transfer Learning** | ⚠️ Conditional | Only if changing architecture |

**Action Plan:**
1. ✅ Test `checkpoint_100000.pt` with diagnostic
2. ✅ If good, resume training with anti-saturation config
3. ✅ Monitor for saturation during training
4. ✅ Adjust hyperparameters if saturation starts

