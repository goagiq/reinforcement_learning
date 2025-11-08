# Training Expectations Guide - What to Expect During Retraining

## 🚀 Quick Start: What You'll See

### Initial Setup (First 30 seconds)
```
✅ Using GPU: [Your GPU Name] (CUDA X.X)
   GPU Memory: X.X GB total
✅ Data loaded successfully (took X.Xs)
Creating trading environment...
Creating PPO agent...
   Using architecture from config/default: [256, 256, 128]
⚙️  Performance mode: [quiet/performance/turbo]
============================================================
Starting Training
============================================================
Device: cuda
Total timesteps: 1,000,000
Timeframes: [1, 5, 15]
Instrument: ES
============================================================
```

### Progress Bar
```
Training: 5%|██▌                    | 50,000/1,000,000 [15:30<4:32:15]
```
- **Shows**: Current timestep, percentage, elapsed time, estimated time remaining
- **Updates**: Continuously as training progresses

---

## 📊 Training Phases & What to Expect

### Phase 1: Initial Exploration (0-15% = 0-150k steps)
**Duration**: ~1-2 hours (depending on GPU)

**What's Happening:**
- Agent is randomly exploring the action space
- Learning basic patterns in the data
- High loss values (normal!)

**Expected Metrics:**
| Metric | Early Phase | End of Phase | What It Means |
|--------|-------------|--------------|---------------|
| **Loss** | 5,000-10,000 | 1,000-3,000 | Decreasing = learning |
| **Policy Loss** | 2,000-5,000 | 500-1,500 | Policy improving |
| **Value Loss** | 3,000-6,000 | 500-1,500 | Value estimates improving |
| **Entropy** | 3.0-4.0 | 2.5-3.5 | Exploration rate (should stabilize) |
| **Episode Reward** | -100 to +50 | -50 to +100 | Random early, improving |
| **Mean Reward (10 eps)** | N/A (need 10 episodes) | -20 to +20 | Starting to learn |

**Console Output:**
```
Episode 10 | Reward: -45.23 | Length: 49,850 | PnL: $-234.50 | Trades: 12
  Last 10 episodes - Mean reward: -38.45, Mean length: 49,823.5
```

**What's Normal:**
- ✅ High loss values (5k-10k) - **This is expected!**
- ✅ Negative rewards - Agent is exploring
- ✅ Loss decreasing slowly
- ✅ Long episodes (49k+ steps) - Normal for large datasets

**Red Flags:**
- ❌ Loss > 50,000 (might indicate data issues)
- ❌ Loss not decreasing at all after 50k steps
- ❌ NaN values in any metric
- ❌ GPU utilization stuck at 0% (if using GPU)

---

### Phase 2: First Profits (15-30% = 150k-300k steps)
**Duration**: ~1-2 hours

**What's Happening:**
- Agent starts recognizing profitable patterns
- Occasional positive rewards
- Mean reward beginning to climb

**Expected Metrics:**
| Metric | Early Phase | End of Phase | What It Means |
|--------|-------------|--------------|---------------|
| **Loss** | 1,000-3,000 | 500-1,500 | Continuing to decrease |
| **Policy Loss** | 500-1,500 | 200-800 | Policy getting better |
| **Value Loss** | 500-1,500 | 200-800 | Better value estimates |
| **Entropy** | 2.5-3.5 | 2.0-3.0 | Still exploring, but more focused |
| **Episode Reward** | -50 to +150 | -20 to +200 | More positive episodes |
| **Mean Reward (10 eps)** | -20 to +20 | 0 to +50 | **Becoming profitable!** |

**Console Output:**
```
Episode 20 | Reward: 87.45 | Length: 49,920 | PnL: $456.78 | Trades: 18
  Last 10 episodes - Mean reward: 23.67, Mean length: 49,901.2
  🎉 New best mean reward: 23.67
```

**What's Normal:**
- ✅ Mix of positive and negative rewards
- ✅ Mean reward trending upward
- ✅ Loss continuing to decrease
- ✅ More trades per episode (agent is more active)

**Red Flags:**
- ❌ Mean reward still negative after 250k steps
- ❌ Loss plateaued (not decreasing)
- ❌ All rewards negative (agent not learning)

---

### Phase 3: Profitability (30-50% = 300k-500k steps)
**Duration**: ~1-2 hours

**What's Happening:**
- Agent consistently finding profitable trades
- Win rate improving
- Risk-adjusted returns getting better

**Expected Metrics:**
| Metric | Early Phase | End of Phase | What It Means |
|--------|-------------|--------------|---------------|
| **Loss** | 500-1,500 | 200-800 | Low and stable |
| **Policy Loss** | 200-800 | 100-400 | Policy well-trained |
| **Value Loss** | 200-800 | 100-400 | Accurate value estimates |
| **Entropy** | 2.0-3.0 | 1.5-2.5 | Less exploration, more exploitation |
| **Episode Reward** | -20 to +200 | 0 to +300 | Mostly positive |
| **Mean Reward (10 eps)** | 0 to +50 | +30 to +100 | **Consistently profitable!** |

**Console Output:**
```
Episode 30 | Reward: 156.78 | Length: 49,950 | PnL: $789.12 | Trades: 24
  Last 10 episodes - Mean reward: 67.45, Mean length: 49,935.8
  🎉 New best mean reward: 67.45

📊 Evaluation @ step 350000: Mean reward: 45.23, Mean PnL: $234.56
```

**What's Normal:**
- ✅ Mean reward consistently positive
- ✅ Evaluation rewards matching training rewards
- ✅ Loss stable and low
- ✅ More consistent performance

**Red Flags:**
- ❌ Mean reward still negative
- ❌ Large variance in rewards (unstable)
- ❌ Evaluation much worse than training (overfitting)

---

### Phase 4: Optimization (50-70% = 500k-700k steps)
**Duration**: ~1-2 hours

**What's Happening:**
- Fine-tuning strategies
- Risk-adjusted returns improving
- Stable performance patterns emerging

**Expected Metrics:**
| Metric | Early Phase | End of Phase | What It Means |
|--------|-------------|--------------|---------------|
| **Loss** | 200-800 | 100-500 | Very low and stable |
| **Policy Loss** | 100-400 | 50-200 | Highly optimized policy |
| **Value Loss** | 100-400 | 50-200 | Very accurate value estimates |
| **Entropy** | 1.5-2.5 | 1.0-2.0 | Focused exploitation |
| **Episode Reward** | 0 to +300 | +50 to +400 | High and consistent |
| **Mean Reward (10 eps)** | +30 to +100 | +80 to +150 | **Highly profitable!** |

**Console Output:**
```
Episode 40 | Reward: 234.56 | Length: 49,980 | PnL: $1,234.56 | Trades: 28
  Last 10 episodes - Mean reward: 123.45, Mean length: 49,967.3
  🎉 New best mean reward: 123.45

📊 Evaluation @ step 600000: Mean reward: 98.76, Mean PnL: $512.34
```

**What's Normal:**
- ✅ High mean rewards
- ✅ Consistent performance
- ✅ Low loss values
- ✅ Good evaluation performance

**Red Flags:**
- ❌ Mean reward decreasing (regression)
- ❌ Loss increasing (overfitting)
- ❌ High variance in rewards

---

### Phase 5: Refinement (70-100% = 700k-1,000k steps)
**Duration**: ~1-2 hours

**What's Happening:**
- Final optimizations
- Best performance metrics
- Ready for deployment

**Expected Metrics:**
| Metric | Early Phase | End of Phase | What It Means |
|--------|-------------|--------------|---------------|
| **Loss** | 100-500 | 50-300 | Minimal loss |
| **Policy Loss** | 50-200 | 25-100 | Optimal policy |
| **Value Loss** | 50-200 | 25-100 | Optimal value estimates |
| **Entropy** | 1.0-2.0 | 0.5-1.5 | Exploitation mode |
| **Episode Reward** | +50 to +400 | +100 to +500 | Maximum performance |
| **Mean Reward (10 eps)** | +80 to +150 | +120 to +200 | **Peak performance!** |

**Console Output:**
```
Episode 50 | Reward: 345.67 | Length: 49,995 | PnL: $1,789.01 | Trades: 32
  Last 10 episodes - Mean reward: 187.89, Mean length: 49,982.1
  🎉 New best mean reward: 187.89

📊 Evaluation @ step 900000: Mean reward: 156.78, Mean PnL: $789.12
```

**What's Normal:**
- ✅ Peak performance metrics
- ✅ Stable, high rewards
- ✅ Low loss values
- ✅ Consistent evaluation

**Red Flags:**
- ❌ Performance plateaued or decreasing
- ❌ Overfitting (training >> evaluation)

---

## 🖥️ GPU Utilization (If Using GPU)

### Turbo Mode (Adaptive)
**What You'll See:**
```
🔥🔥🔥 TURBO MODE ACTIVE (ADAPTIVE) 🔥🔥🔥
   Episode: 1
   Batch size: 6400 (50.00x base: 128)
   Epochs: 240 (8.00x base: 30)
   Target: 65% GPU, <8GB VRAM
   📊 Peak GPU during update: 72.3% (sampled during update)
   📈 GPU too low (45.2% < 65%), increasing: batch 50.00x → 75.00x (+50%)
```

**Expected Behavior:**
- GPU utilization: 50-80% (target: 65%)
- VRAM usage: 2-8GB (target: <8GB)
- Batch size: Automatically adjusts (2x to 100x base)
- Epochs: Automatically adjusts (1.5x to 15x base)

**What's Normal:**
- ✅ GPU utilization increasing over time (system optimizing)
- ✅ VRAM usage stable
- ✅ Batch size/epochs adjusting automatically

**Red Flags:**
- ❌ GPU utilization stuck at 0% (not using GPU)
- ❌ VRAM > 90% (might cause OOM)
- ❌ Batch size not adjusting (Turbo not working)

### Performance Mode
**What You'll See:**
```
⚙️  Performance mode: performance
   Batch size: 256 (2x base: 128)
   Epochs: 45 (1.5x base: 30)
```

**Expected Behavior:**
- GPU utilization: 30-60%
- VRAM usage: 1-4GB
- Fixed multipliers (2x batch, 1.5x epochs)

### Quiet Mode (Default)
**What You'll See:**
```
⚙️  Performance mode: quiet
   Batch size: 128 (base: 128)
   Epochs: 30 (base: 30)
```

**Expected Behavior:**
- GPU utilization: 20-40%
- VRAM usage: 1-3GB
- Base settings (no multipliers)

---

## 📈 Episode Progress

### Episode Length
**Expected:** 49,000-50,000 steps per episode (for typical NT8 data)

**Why So Long?**
- Your data has many bars (e.g., 50,000 bars)
- Episode = total_bars - lookback_bars - 1
- This is **normal** for large datasets

**Console Output:**
```
Episode 10 | Reward: 87.45 | Length: 49,850 | PnL: $456.78 | Trades: 18
```
- **Length**: Number of steps in episode
- **Reward**: Cumulative reward for episode
- **PnL**: Profit/Loss in dollars
- **Trades**: Number of trades executed

### Episode Frequency
- **First episode**: Takes ~50k steps (one full pass through data)
- **Subsequent episodes**: Same length (data resets)
- **Episode prints**: Every 10 episodes (episodes 10, 20, 30, etc.)

---

## 💾 Checkpoint Saving

### Checkpoint Frequency
- **Every 10,000 steps** (as configured)
- **Best model**: Saved when mean reward improves
- **Final model**: Saved at end of training

### Checkpoint Files
```
models/
  checkpoint_10000.pt
  checkpoint_20000.pt
  checkpoint_30000.pt
  ...
  best_model.pt          # Best performing model
  final_model.pt         # Final trained model
```

### Checkpoint Size
- **Typical**: 3-5 MB per checkpoint
- **Contains**: Model weights, optimizer states, training progress

---

## 🎯 Evaluation Runs

### Evaluation Frequency
- **Every 5,000 steps** (as configured)
- **Runs 5 episodes** with deterministic actions (no exploration)

### Evaluation Output
```
📊 Evaluation @ step 50000: Mean reward: 23.45, Mean PnL: $123.45
📊 Evaluation @ step 100000: Mean reward: 45.67, Mean PnL: $234.56
```

**What to Expect:**
- **Early training**: Evaluation reward < training reward (normal - agent exploring)
- **Mid training**: Evaluation reward ≈ training reward (good - no overfitting)
- **Late training**: Evaluation reward ≈ training reward (excellent - stable)

**Red Flags:**
- ❌ Evaluation reward << training reward (overfitting)
- ❌ Evaluation reward decreasing (regression)

---

## ⏱️ Timeline Expectations

### Total Training Time
**For 1,000,000 steps:**

| Device | Mode | Estimated Time |
|--------|------|----------------|
| **GPU (Turbo)** | Adaptive | 4-6 hours |
| **GPU (Performance)** | 2x batch | 6-8 hours |
| **GPU (Quiet)** | Base | 8-12 hours |
| **CPU** | Base | 24-48 hours |

**Note:** Times vary based on:
- GPU model (RTX 3060 vs RTX 4090)
- Data size (number of bars)
- Episode length
- System load

### Milestone Timeline

| Timesteps | % Complete | Expected Time | What's Happening |
|-----------|------------|---------------|------------------|
| 0-50k | 0-5% | 15-30 min | Initial exploration |
| 50k-150k | 5-15% | 1-2 hours | Learning patterns |
| 150k-300k | 15-30% | 2-3 hours | First profits |
| 300k-500k | 30-50% | 3-4 hours | Becoming profitable |
| 500k-700k | 50-70% | 4-5 hours | Optimization |
| 700k-1M | 70-100% | 5-6 hours | Refinement |

---

## ✅ Health Check Indicators

### Good Training Signs ✅
- ✅ Loss decreasing over time
- ✅ Mean reward trending upward
- ✅ GPU utilization > 20% (if using GPU)
- ✅ No NaN values
- ✅ Checkpoints saving regularly
- ✅ Evaluation rewards improving
- ✅ Progress bar advancing steadily

### Warning Signs ⚠️
- ⚠️ Loss plateaued (not decreasing for 50k+ steps)
- ⚠️ Mean reward still negative after 300k steps
- ⚠️ GPU utilization stuck at 0% (if using GPU)
- ⚠️ VRAM > 90% (might cause OOM)
- ⚠️ Evaluation much worse than training (overfitting)

### Critical Issues ❌
- ❌ NaN values in any metric
- ❌ Training stopped/crashed
- ❌ Loss > 100,000 (data issues)
- ❌ No checkpoints saving
- ❌ GPU errors (CUDA out of memory, etc.)

---

## 📊 TensorBoard Monitoring (If Enabled)

### Accessing TensorBoard
```bash
tensorboard --logdir logs
# Then open http://localhost:6006
```

### What to Monitor

**1. Training Loss (`train/loss`)**
- Should decrease over time
- May have spikes (normal)
- Should stabilize in later phases

**2. Policy Loss (`train/policy_loss`)**
- Should decrease over time
- Lower = better policy

**3. Value Loss (`train/value_loss`)**
- Should decrease over time
- Lower = better value estimates

**4. Entropy (`train/entropy`)**
- Should stabilize around 1.5-3.0
- Decreasing = less exploration

**5. Episode Reward (`episode/reward`)**
- Should trend upward
- May have variance (normal)

**6. Evaluation Reward (`eval/mean_reward`)**
- Should match training reward
- Should improve over time

---

## 🎓 Understanding the Metrics

### Loss
- **What it is**: How wrong the agent's predictions are
- **Good**: Decreasing over time
- **Bad**: Increasing or plateaued
- **Normal range**: 50-5,000 (depends on phase)

### Policy Loss
- **What it is**: How much the policy needs to change
- **Good**: Decreasing (policy improving)
- **Bad**: Increasing (policy getting worse)
- **Normal range**: 25-2,000 (depends on phase)

### Value Loss
- **What it is**: How wrong value estimates are
- **Good**: Decreasing (better value estimates)
- **Bad**: Increasing (worse value estimates)
- **Normal range**: 25-2,000 (depends on phase)

### Entropy
- **What it is**: Exploration rate (randomness)
- **Good**: Stable around 1.5-3.0
- **Bad**: Too low (<0.5) = no exploration, too high (>5.0) = too random
- **Normal range**: 0.5-4.0

### Episode Reward
- **What it is**: Cumulative reward for an episode
- **Good**: Increasing over time, becoming positive
- **Bad**: Always negative, decreasing
- **Normal range**: -100 to +500 (depends on phase)

### Mean Reward (10 episodes)
- **What it is**: Average reward over last 10 episodes
- **Good**: Trending upward, becoming positive
- **Bad**: Stuck negative, decreasing
- **Normal range**: -50 to +200 (depends on phase)

---

## 🔧 Troubleshooting

### Training Too Slow?
1. **Enable Turbo Mode**: Set `turbo_training_mode: true` in `settings.json`
2. **Use GPU**: Ensure `device: "cuda"` in config
3. **Increase batch size**: Edit `batch_size` in config (if VRAM allows)
4. **Reduce data**: Use smaller timeframes or fewer bars

### Loss Not Decreasing?
1. **Check learning rate**: Too high (unstable) or too low (slow)
2. **Check data**: Ensure data is loaded correctly
3. **Check rewards**: If rewards are all negative, agent can't learn
4. **Wait longer**: Some phases take time

### GPU Not Being Used?
1. **Check CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
2. **Check config**: `device: "cuda"` in config
3. **Check GPU**: `nvidia-smi` to see if GPU is active

### Out of Memory (OOM)?
1. **Reduce batch size**: Lower `batch_size` in config
2. **Disable Turbo**: Use `quiet` mode
3. **Reduce data**: Use smaller datasets
4. **Close other programs**: Free up GPU memory

---

## 📝 Summary: What to Watch For

### First 30 Minutes
- ✅ Training starts successfully
- ✅ Progress bar advancing
- ✅ Loss values appearing (high is normal)
- ✅ GPU utilization > 0% (if using GPU)

### First Hour
- ✅ Loss decreasing
- ✅ First episode completes (if data is small)
- ✅ Episode counter increments
- ✅ Episode rewards appearing (negative is normal)

### First 2 Hours
- ✅ Mean reward trending upward
- ✅ Loss continuing to decrease
- ✅ More positive episode rewards
- ✅ Evaluation rewards improving

### After 3 Hours
- ✅ Mean reward becoming positive
- ✅ Consistent performance
- ✅ Low loss values
- ✅ Good evaluation performance

### Final Hours
- ✅ Peak performance metrics
- ✅ Stable, high rewards
- ✅ Low loss values
- ✅ Ready for deployment

---

## 🎉 Success Criteria

**Training is successful if:**
1. ✅ Loss decreased significantly from start
2. ✅ Mean reward is positive (after 300k+ steps)
3. ✅ Evaluation rewards match training rewards
4. ✅ No NaN values or crashes
5. ✅ Checkpoints saving regularly
6. ✅ GPU utilization reasonable (if using GPU)
7. ✅ Final model saved successfully

**Model is ready for deployment if:**
1. ✅ Mean reward > 50 (after 500k+ steps)
2. ✅ Evaluation reward > 40
3. ✅ Loss < 500
4. ✅ Consistent performance over last 100k steps
5. ✅ No overfitting (eval ≈ training)

---

**Good luck with your training! 🚀**

