# Episode Length Configuration Guide

## Overview

Episodes in RL training represent complete trading sessions. To ensure proper learning and metric tracking, episodes must complete in a reasonable time. This guide explains episode length management.

## The Problem

Your historical data may contain **85,000+ time steps** across multiple timeframes. If episodes tried to use the entire dataset:
- Episodes would take hours/days to complete
- No episode metrics would be recorded (all stuck in progress)
- Training progress difficult to track
- Agent updates would be infrequent

## The Solution: Episode Length Limit

### Configuration

In `configs/train_config.yaml`:

```yaml
environment:
  max_episode_steps: 10000  # Episodes complete after this many steps
```

**Default: 10,000 steps** - This ensures episodes complete regularly while providing enough context for learning.

### How It Works

1. **Episode Starts**: Environment resets, agent starts trading
2. **Episode Progress**: Agent takes actions, receives rewards
3. **Episode Completes**: When `current_step >= max_episode_steps` (default: 10,000)
4. **Episode Resets**: Environment resets, new episode begins
5. **Metrics Recorded**: Episode reward, length, PnL, trades tracked

### Episode vs Data Length

- **Data Length**: 85,747 steps (your historical data)
- **Episode Length**: 10,000 steps (configurable limit)
- **Episodes per Dataset**: ~8.5 episodes can be created from full dataset

**The environment can:**
- ✅ Start each episode from different points in the data
- ✅ Use the same data segment multiple times
- ✅ Create multiple episodes from one dataset
- ✅ Complete episodes regularly for proper learning

## Choosing Episode Length

### Default: 10,000 Steps (Recommended)

**Pros:**
- ✅ Episodes complete in reasonable time (~minutes)
- ✅ Good balance of context and training speed
- ✅ Works well with PPO's update frequency (every 2,048 steps)
- ✅ Allows 4-5 updates per episode

**Use when:**
- General training scenarios
- Standard dataset sizes
- Balanced training speed/context needs

### Short Episodes: 5,000-7,000 Steps

**Pros:**
- ✅ Faster episode completion
- ✅ More episodes per training run
- ✅ Quicker iteration cycles
- ✅ Better for early exploration

**Cons:**
- ⚠️ Less context per episode
- ⚠️ May miss longer-term patterns

**Use when:**
- Rapid experimentation
- Testing different configurations
- Limited training time

### Long Episodes: 15,000-20,000 Steps

**Pros:**
- ✅ More context per episode
- ✅ Can capture longer-term trends
- ✅ Better for strategies that need more time

**Cons:**
- ⚠️ Slower episode completion
- ⚠️ Fewer episodes per training session
- ⚠️ Longer time to see progress

**Use when:**
- Training on very long-term patterns
- Need more historical context
- Patient with training time

### No Limit (Full Data Length)

**Not Recommended** for very long datasets:
- Episodes may never complete
- No metrics tracking
- Poor training dynamics

**Only use if:**
- Dataset is small (< 10,000 steps)
- You specifically need full-dataset episodes

## Implementation Details

### Environment Code

```python
# src/trading_env.py
def __init__(self, ..., max_episode_steps=None):
    # Calculate available steps from data
    data_max_steps = len(data) - lookback_bars - 1
    
    # Use configured limit if provided and smaller than data
    self.max_steps = (
        max_episode_steps 
        if max_episode_steps is not None and max_episode_steps < data_max_steps
        else data_max_steps
    )
```

### Training Code

```python
# src/train.py
max_episode_steps = config["environment"].get("max_episode_steps", 10000)

env = TradingEnvironment(
    data=multi_tf_data,
    max_episode_steps=max_episode_steps  # Applied here
)
```

### Episode Completion

```python
# In environment step()
terminated = self.current_step >= self.max_steps
```

When `terminated=True`, the episode ends and resets.

## Monitoring Episode Completion

### In Console

Look for debug messages:
```
[DEBUG] TradingEnvironment: current_step=10000, max_steps=10000, terminated=True
[DEBUG] Episode completing: length=10000, reward=-22.07
```

### In UI

Check metrics:
- **Current Episode**: Should increment regularly
- **Completed Episodes**: Should increase over time
- **Episode Length**: Should stabilize around configured limit

### Expected Behavior

After configuration:
- ✅ Episodes complete at `max_episode_steps`
- ✅ `completed_episodes` counter increments
- ✅ `mean_reward_10` updates with completed episodes
- ✅ Episode rewards/lengths tracked properly

## Common Issues and Fixes

### Issue: Episodes Never Complete

**Symptoms:**
- `completed_episodes: 0` for extended period
- `current_episode_length` keeps growing beyond 10,000
- No episode metrics updating

**Causes:**
1. `max_episode_steps` not set in config
2. Config value larger than data length
3. Environment not using configured value

**Fixes:**
1. Add `max_episode_steps: 10000` to config
2. Restart training (or resume from checkpoint)
3. Verify in console: "Max episode steps: 10000"

### Issue: Episodes Complete Too Quickly

**Symptoms:**
- Episodes completing in < 1000 steps
- Not enough context per episode

**Fixes:**
- Increase `max_episode_steps` to 15,000-20,000
- Check if data is being reset incorrectly

### Issue: Episodes Take Too Long

**Symptoms:**
- Episodes taking hours to complete
- Slow training progress

**Fixes:**
- Decrease `max_episode_steps` to 5,000-7,000
- Consider shorter datasets for initial training

## Relationship to Training Progress

### Timesteps vs Episodes

- **Timesteps**: Total number of training steps (across all episodes)
- **Episodes**: Number of completed trading sessions

**Example:**
- 460,000 timesteps = ~46 completed episodes (at 10K steps each)
- Each episode contributes to learning
- More episodes = more diverse experiences

### PPO Update Frequency

PPO updates every `n_steps` (default: 2,048 steps):
- At 10,000 episode length: ~5 updates per episode
- Provides good learning signal
- Balances stability and efficiency

### Checkpoint Frequency

Checkpoints saved every `save_freq` timesteps (default: 10,000):
- Every checkpoint = ~1 completed episode
- Allows resuming with minimal data loss
- Episodes complete between checkpoints

## Best Practices

1. **Start with Default**: Use 10,000 steps initially
2. **Monitor Completion**: Verify episodes complete regularly
3. **Adjust if Needed**: Modify based on training behavior
4. **Keep Reasonable**: Avoid extremes (< 1K or > 50K)
5. **Consider Data**: Shorter datasets can use full length

## Configuration Example

```yaml
# configs/train_config.yaml
environment:
  max_episode_steps: 10000  # Default - good balance
  # max_episode_steps: 5000   # Faster episodes
  # max_episode_steps: 20000  # More context
  # max_episode_steps: null   # Use full data (not recommended for long datasets)
```

## Related Documentation

- `docs/TRAINING_CONFIGURATION.md` - Complete training config guide
- `docs/REWARD_FUNCTION.md` - How rewards work during episodes
- `docs/HOW_RL_TRADING_WORKS.md` - Overall training explanation
- `configs/train_config.yaml` - Configuration file

