# Training Configuration Guide

## Overview

This guide explains the training configuration options and how they affect the RL agent's learning process.

## Configuration File

Main configuration: `configs/train_config.yaml`

## Key Configuration Sections

### 1. Environment Configuration

#### Episode Length Management

```yaml
environment:
  max_episode_steps: 10000  # Maximum steps per episode
```

**Why this matters:**
- Your historical data may contain 85,000+ time steps
- Without a limit, episodes would never complete, preventing proper learning
- Default: 10,000 steps ensures episodes complete in reasonable time
- Episodes automatically restart after reaching this limit

**Adjusting:**
- **Longer episodes** (15,000-20,000): More context per episode, slower training
- **Shorter episodes** (5,000-7,000): Faster episode completion, less context
- **No limit** (set to `null`): Use full data length (not recommended for very long datasets)

#### Timeframes

```yaml
timeframes: [1, 5, 15]  # Minutes
```

- **1min**: Short-term momentum, entry/exit timing
- **5min**: Medium-term trends, swing opportunities
- **15min**: Overall market direction, major trends

#### Lookback Bars

```yaml
lookback_bars: 20  # Historical bars included in state
```

- Controls how much historical context the agent sees
- More bars = more context but higher state dimensionality
- Typical range: 15-30 bars

### 2. Reward Function Configuration

```yaml
reward:
  pnl_weight: 1.0              # Primary reward signal (PnL changes)
  transaction_cost: 0.0001     # Transaction cost per trade
  risk_penalty: 0.5            # Risk penalty coefficient (applied at 10% effective weight)
  drawdown_penalty: 0.3        # Drawdown penalty (applied at 10% effective weight)
```

**Current Implementation:**
- **Effective risk_penalty**: 0.05 (0.5 * 0.1) - Reduced to allow learning
- **Effective drawdown_penalty**: 0.03 (0.3 * 0.1) - Only applies if DD > 15%
- **Holding cost**: 0.0000001 per step (0.1% of transaction cost)
- **Profit bonus**: +10% bonus on positive PnL changes
- **Scaling**: 10x multiplier for gradient stability

See `docs/REWARD_FUNCTION.md` for detailed explanation.

### 3. Model Configuration (PPO)

```yaml
model:
  algorithm: "PPO"
  learning_rate: 0.0003       # Adam optimizer learning rate
  batch_size: 64              # Increase if GPU available (128-256)
  n_steps: 2048               # Steps per update (rollout length)
  gamma: 0.99                  # Discount factor (how much future rewards matter)
  gae_lambda: 0.95            # GAE lambda (advantage estimation)
  clip_range: 0.2             # PPO clip range (prevents large policy updates)
  value_loss_coef: 0.5        # Value function loss coefficient
  entropy_coef: 0.01          # Entropy bonus (encourages exploration)
  max_grad_norm: 0.5          # Gradient clipping threshold
```

**Tuning Guidelines:**
- **Learning rate**: Lower (1e-4) for stability, higher (1e-3) for faster learning
- **Batch size**: Increase with GPU memory (64 → 128 → 256)
- **n_steps**: Longer rollouts (4096) for more stable updates
- **Entropy**: Increase (0.02-0.05) if agent is too conservative

### 4. Training Configuration

```yaml
training:
  total_timesteps: 1000000     # Total training steps across all episodes
  save_freq: 10000             # Save checkpoint every N timesteps
  eval_freq: 5000              # Run evaluation every N timesteps
  device: "cuda"               # "cuda" or "cpu"
```

**Training Progress:**
- Episodes complete at `max_episode_steps` (default: 10,000)
- Each episode contributes to learning
- Metrics tracked: episode rewards, lengths, PnL, trades, win rate
- Checkpoints allow resuming without losing progress

## Episode Structure

### How Episodes Work

1. **Episode Start**: Environment resets, agent starts at beginning of data
2. **Episode Steps**: Agent takes actions, receives rewards (up to `max_episode_steps`)
3. **Episode End**: Reaches step limit OR data end
4. **Episode Reset**: Environment resets, starts new episode (may reuse same data segment)

### Episode Metrics

After each episode completion:
- Episode reward (cumulative)
- Episode length (steps)
- Episode PnL (profit/loss)
- Number of trades
- Win rate

### Mean Metrics

- **Mean Reward (last 10)**: Average reward across last 10 completed episodes
- **Mean Episode Length**: Average episode length across all completed episodes
- **Best Reward**: Highest episode reward seen so far

## Resuming Training

### From Checkpoint

Checkpoints save:
- ✅ Agent weights (actor & critic networks)
- ✅ Optimizer states (training momentum)
- ✅ Training progress (timestep, episode count)
- ✅ Episode history (rewards, lengths)

**To resume:**
```python
# In UI: Select latest checkpoint when starting training
# Or via API:
{
  "checkpoint_path": "models/checkpoint_460000.pt"
}
```

**Benefits:**
- Preserves all learned knowledge
- Continues from exact timestep
- No loss of training progress

### When to Resume vs Restart

**Resume (Recommended):**
- ✅ Want to continue existing training
- ✅ Agent has learned valuable patterns
- ✅ Save training time

**Restart:**
- ✅ Want clean metrics from start
- ✅ Changed environment/reward function significantly
- ✅ Testing different hyperparameters

## Training Monitoring

### Real-Time Metrics (UI)

The training UI shows:
- **Current Episode**: Episode number (completed + in-progress)
- **Timesteps**: Total steps trained so far
- **Latest Reward**: Current/latest episode reward
- **Mean Reward (Last 10)**: Average of recent episodes
- **Training Loss**: Policy and value function loss
- **Entropy**: Exploration vs exploitation balance

### Console Output

During training, you'll see:
```
Episode 10 | Reward: -22.07 | Length: 10000 | PnL: $-1250.50 | Trades: 15
  Last 10 episodes - Mean reward: -24.32, Mean length: 10000.0
```

### Debug Messages

If episodes are near completion, debug messages appear:
```
[DEBUG] TradingEnvironment: current_step=9999, max_steps=10000, terminated=True
[DEBUG] Episode completing: length=10000, reward=-22.07, terminated=True
```

## Common Issues

### Episodes Not Completing

**Symptom:** `completed_episodes=0` even after many timesteps

**Cause:** Episode length limit not set or too high

**Fix:** Ensure `max_episode_steps: 10000` in config (or appropriate value)

### Rewards Always Negative

**Symptom:** No positive episode rewards observed

**Causes:**
1. Early in training (normal - agent is learning)
2. Reward function too harsh (penalties too high)
3. Agent hasn't learned profitable patterns yet

**Expected Timeline:**
- 0-100K timesteps: Mostly negative (exploration)
- 100K-500K: Mixed rewards (learning)
- 500K+: Should see positive rewards if agent is learning

**Fix:** Adjust reward function parameters (see `docs/REWARD_FUNCTION.md`)

### Episodes Too Long/Short

**Symptom:** Episodes complete too quickly or take forever

**Fix:** Adjust `max_episode_steps` in config:
- Too short: Increase to 15,000-20,000
- Too long: Decrease to 5,000-7,000

### Training Slow

**Causes:**
1. Using CPU instead of GPU
2. Batch size too small
3. Too many epochs per update

**Fixes:**
- Use `device: "cuda"` in config
- Increase `batch_size` if GPU available
- Reduce `n_epochs` if updates are slow

## Performance Optimization

### GPU Training

```yaml
training:
  device: "cuda"  # Use GPU
model:
  batch_size: 128  # Increase for GPU
```

### Performance Mode

Set in `settings.json`:
```json
{
  "performance_mode": "performance"
}
```

**Effects:**
- Doubles batch size
- Increases training epochs by 50%
- Faster learning at cost of higher resource usage

### Quiet Mode (Default)

```json
{
  "performance_mode": "quiet"
}
```

**Effects:**
- Standard batch size and epochs
- Lower resource usage
- Good for long-running training

## Best Practices

1. **Start Small**: Begin with 100K-200K timesteps to test setup
2. **Monitor Progress**: Watch mean reward trends over time
3. **Save Checkpoints**: Enable regular checkpoint saving
4. **Resume Smart**: Continue from checkpoints unless major changes
5. **Adjust Gradually**: Change one hyperparameter at a time
6. **Use GPU**: Significantly faster training with CUDA

## Configuration Examples

### Fast Training (GPU)

```yaml
model:
  batch_size: 256
  n_steps: 4096
training:
  device: "cuda"
environment:
  max_episode_steps: 10000
```

### Stable Training (CPU)

```yaml
model:
  batch_size: 64
  learning_rate: 0.0002
training:
  device: "cpu"
environment:
  max_episode_steps: 10000
```

### Conservative Learning

```yaml
model:
  learning_rate: 0.0001
  entropy_coef: 0.02
environment:
  max_episode_steps: 15000
```

## Related Documentation

- `docs/REWARD_FUNCTION.md` - Detailed reward function explanation
- `docs/HOW_RL_TRADING_WORKS.md` - High-level training explanation
- `docs/RESUME_TRAINING.md` - Resume training guide
- `configs/train_config.yaml` - Full configuration file

