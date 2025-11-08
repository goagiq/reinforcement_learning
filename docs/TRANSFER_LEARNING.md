# Transfer Learning Guide

## Overview

This guide explains how to use transfer learning when your RL model architecture changes (e.g., expanding from [128, 128, 64] to [256, 256, 128]).

## What is Transfer Learning?

Transfer learning allows you to preserve learned knowledge from a trained model when changing the architecture. Instead of starting training from scratch, you can:

1. **Preserve learned weights** from the old architecture
2. **Transfer compatible weights** to the new architecture
3. **Initialize new dimensions** intelligently to avoid disrupting learned patterns

## When to Use Transfer Learning

✅ **Use transfer learning when:**
- You want to expand the model architecture (e.g., [128, 128, 64] → [256, 256, 128])
- You have a well-trained model with valuable learned patterns
- You want to continue training with a larger model without losing progress

❌ **Don't use transfer learning when:**
- The state dimension changes (input features change)
- You want to start completely fresh
- The architecture change is too drastic (e.g., completely different network structure)

## How It Works

### Architecture Mapping

The transfer learning system maps weights from old to new architecture:

**Example: [128, 128, 64] → [256, 256, 128]**

1. **Layer 1 (900 → 128 → 900 → 256)**:
   - Copies first 128 neurons' weights
   - Initializes remaining 128 neurons with small random values (10% scale)

2. **Layer 2 (128 → 128 → 256 → 256)**:
   - Copies first 128×128 block of weights
   - Initializes remaining dimensions with small random values

3. **Layer 3 (128 → 64 → 256 → 128)**:
   - Copies first 64×128 block of weights
   - Initializes remaining dimensions with small random values

4. **Output Heads (64 → 1 → 128 → 1)**:
   - Copies first 64 rows of weights
   - Initializes remaining rows with small random values

### Transfer Strategies

Three strategies are available:

1. **`copy_and_extend`** (Recommended):
   - Copies old weights exactly
   - Initializes new dimensions with small random values (10% of old weight scale)
   - **Best for**: Preserving learned patterns while allowing new capacity

2. **`interpolate`**:
   - Copies old weights
   - Fills new dimensions by interpolating from existing neurons
   - **Best for**: When you want new neurons to start similar to existing ones

3. **`zero_pad`**:
   - Copies old weights
   - Initializes new dimensions with zeros
   - **Best for**: Conservative approach (new neurons start inactive)

## Usage

### Automatic Detection

The training system **automatically detects architecture mismatches** and uses transfer learning:

```bash
python src/train.py --config configs/train_config.yaml --checkpoint models/best_model.pt
```

When you resume training with a different architecture, you'll see:

```
⚠️  Architecture mismatch detected!
   Checkpoint: state_dim=900, hidden_dims=[128, 128, 64]
   Current:    state_dim=900, hidden_dims=[256, 256, 128]
   🔄 Using transfer learning to preserve learned knowledge...
```

### Manual Transfer

You can also manually transfer weights programmatically:

```python
from src.rl_agent import PPOAgent
from src.weight_transfer import transfer_checkpoint_weights

# Create new agent with larger architecture
agent = PPOAgent(
    state_dim=900,
    hidden_dims=[256, 256, 128],  # New architecture
    device="cuda"
)

# Transfer weights from old checkpoint
agent.load_with_transfer(
    "models/best_model.pt",
    transfer_strategy="copy_and_extend"
)
```

### Configuring Transfer Strategy

Add to your training config:

```yaml
training:
  transfer_strategy: "copy_and_extend"  # Options: copy_and_extend, interpolate, zero_pad
```

## Example: Expanding Architecture

### Step 1: Original Training

```yaml
# configs/train_config_old.yaml
model:
  hidden_dims: [128, 128, 64]  # Smaller architecture
```

Train model → saves as `models/best_model.pt` with architecture [128, 128, 64]

### Step 2: Update Configuration

```yaml
# configs/train_config.yaml
model:
  hidden_dims: [256, 256, 128]  # Larger architecture
```

### Step 3: Resume Training

```bash
python src/train.py \
  --config configs/train_config.yaml \
  --checkpoint models/best_model.pt \
  --device cuda
```

The system will:
1. Detect architecture mismatch
2. Automatically transfer weights
3. Continue training from transferred checkpoint

## Best Practices

### 1. Start with Lower Learning Rate

After transfer learning, use a slightly lower learning rate for the first few updates to allow the new dimensions to adapt:

```yaml
model:
  learning_rate: 0.0003  # Reduced from 0.0005
```

### 2. Monitor Training Metrics

Watch for:
- **Initial performance drop**: Normal, new dimensions need to learn
- **Rapid recovery**: Good sign, transfer learning worked
- **Stable improvement**: Ideal, model is leveraging new capacity

### 3. Use Appropriate Strategy

- **`copy_and_extend`**: Default, good for most cases
- **`interpolate`**: When you want new neurons to be similar to existing ones
- **`zero_pad`**: Conservative, new neurons start inactive

### 4. Verify State Dimension Matches

⚠️ **Important**: Transfer learning only works when `state_dim` matches!

- ✅ Same state_dim (900) → Transfer learning works
- ❌ Different state_dim → Transfer learning fails (must retrain)

## Troubleshooting

### Error: "State dimension mismatch"

**Problem**: Checkpoint has different `state_dim` than current config.

**Solution**: 
- Check your `environment.state_features` or `lookback_bars` settings
- Ensure state dimension calculation matches: `15 features/tf × 3 timeframes × 20 lookback = 900`

### Error: "Could not transfer optimizer states"

**Problem**: Optimizer states can't be transferred between architectures.

**Solution**: This is normal! Optimizers are automatically reinitialized for the new architecture.

### Poor Performance After Transfer

**Problem**: Model performance drops after transfer learning.

**Possible Causes**:
1. New dimensions initialized too large → Reduce initialization scale
2. Learning rate too high → Lower learning rate temporarily
3. Architecture change too drastic → Consider smaller expansion

**Solutions**:
1. Use `zero_pad` strategy for more conservative transfer
2. Lower learning rate for first 10k steps
3. Gradually expand architecture (e.g., [128, 128, 64] → [192, 192, 96] → [256, 256, 128])

## Technical Details

### Weight Transfer Algorithm

For each layer:

1. **Copy compatible weights**: 
   ```python
   new_weight[:old_out_dim, :old_in_dim] = old_weight[:old_out_dim, :old_in_dim]
   ```

2. **Initialize new output dimensions**:
   ```python
   scale = old_weight.std() * 0.1  # 10% of original scale
   new_weight[old_out_dim:, :old_in_dim] = torch.randn(...) * scale
   ```

3. **Initialize new input dimensions**:
   ```python
   new_weight[:, old_in_dim:] = torch.randn(...) * scale
   ```

### Preserved Information

✅ **Preserved**:
- Learned feature representations
- Policy patterns
- Value estimates
- Training progress (timestep, episode, metrics)

❌ **Not Preserved**:
- Optimizer states (reinitialized)
- Exact weight values for new dimensions (initialized fresh)

## References

- [Training Configuration Guide](TRAINING_CONFIGURATION.md)
- [Resume Training Guide](RESUME_TRAINING.md)
- [Model Architecture](models.py)

