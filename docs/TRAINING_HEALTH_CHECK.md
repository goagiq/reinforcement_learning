# Training Health Check Guide

## ✅ What I See (Everything is Healthy!)

### Current Status:
```
Status: "running" ✅
Timesteps: 10,000 / 1,000,000 (1.0%) ✅
Training Metrics: Updating ✅
  - Loss: 6,973.94 ✅
  - Policy Loss: 3,818.07 ✅
  - Value Loss: 6,311.80 ✅
  - Entropy: 3.41 ✅
```

### Episode Counter Behavior (Understanding Why it's 0):

**Episode Definition:**
- An episode ends when `current_step >= max_steps`
- Your data has `max_steps = total_bars - lookback_bars - 1`

**Why Episode = 0 Still:**
- The agent is **mid-episode** at step 10,000
- First episode hasn't completed yet
- This is **NORMAL** for continuous training

**When Episodes Increment:**
1. When agent reaches end of data
2. Environment resets automatically
3. `episode` counter increments
4. Episode rewards/lengths are recorded

---

## 🔍 How to Verify Training is Healthy

### 1. **Check API Returns 200** ✅
```bash
curl http://localhost:8200/api/training/status
```
You confirmed this returns 200 OK.

### 2. **Check Training Metrics Are Updating** ✅
```json
{
  "training_metrics": {
    "loss": 6973.94,         // Should decrease over time
    "policy_loss": 3818.07,   // Should decrease over time
    "value_loss": 6311.80,    // Should decrease over time
    "entropy": 3.41           // Should stabilize
  }
}
```
**These are updating, which means training is working!**

### 3. **Expected Behavior:**

| Metric | What to Expect | Current Status |
|--------|----------------|----------------|
| **Timestep** | Increases steadily | ✅ 10k → 1M |
| **Progress %** | 0% → 100% | ✅ 1.0% |
| **Episode** | Stays at 0 for a while, then increments | ✅ 0 (normal) |
| **Loss** | Starts high, decreases over time | ✅ 6,973 (high is normal) |
| **Reward** | Random early on, improves with time | ✅ 0.0 (normal) |

### 4. **First Episode Takes Time:**

**Episode Length Depends on:**
- Size of your NT8 data file
- Lookback window (20 bars)
- How many bars you exported

**Example:**
- If you exported 50,000 bars of 1-minute data
- Episode length ≈ 49,979 steps (50,000 - 20 - 1)
- At current rate, first episode takes many timesteps!

---

## 📊 What Console Should Show

### Initial Print:
```
============================================================
Starting Training
============================================================
Device: cuda
Total timesteps: 1,000,000
Timeframes: [1, 5, 15]
Instrument: ES
============================================================

Loaded data for ES with timeframes: [1, 5, 15]
  1min: XXX bars, from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS
  5min: XXX bars, from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS
  15min: XXX bars, from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS
```

### Progress Prints:
```
Episode 10 | Reward: X.XX | Length: XXX | PnL: $XXX.XX | Trades: XX
  Last 10 episodes - Mean reward: X.XX, Mean length: XXX.X

Episode 20 | Reward: X.XX | Length: XXX | PnL: $XXX.XX | Trades: XX
  Last 10 episodes - Mean reward: X.XX, Mean length: XXX.X
```

**You won't see episode prints until Episode 10, 20, 30, etc.**

---

## ⚠️ Warning Signs (None of These Apply to You)

### ❌ Training NOT Working:
- `status: "error"` (You have `"running"`) ✅
- Metrics stuck at same values for minutes (Yours are updating) ✅
- `thread.is_alive()` returns `False` (Training would stop) ✅
- Console shows Python errors (No errors reported) ✅

### ❌ System Issues:
- GPU not detected (You're using CUDA) ✅
- Data file missing (Training started successfully) ✅
- NaN in loss values (Losses are real numbers) ✅

---

## 🎯 What to Look For As Training Continues

### Immediate (First Few Minutes):
- ✅ Progress % should increase: 1% → 2% → 3%
- ✅ Timestep should increase: 10k → 20k → 30k
- ✅ Training metrics should update (working!)

### Soon (Within 30 Minutes):
- ✅ First episode should complete
- ✅ Episode counter increments to 1
- ✅ `latest_reward` and `mean_reward_10` show non-zero values
- ✅ Console prints "Episode 10..."

### Over Time (Hours):
- 📈 Loss values should **decrease**
- 📈 Rewards should **increase** (become positive)
- 📈 Win rate improves
- 📈 PnL becomes positive

---

## 📋 Health Check Checklist

```bash
# 1. Check API is responding
curl http://localhost:8200/api/training/status

# 2. Check status is "running"
# → ✅ You confirmed this

# 3. Check metrics are updating
# Refresh UI, metrics should change
# → ✅ Losses updated

# 4. Check GPU is being used
# Look for "Using GPU: [Your GPU Name]" in console
# → ✅ You confirmed GPU training earlier

# 5. Check data loaded successfully
# Look for "Loaded data for ES..." in console
# → ✅ Training started, so data loaded

# 6. Check progress is advancing
# Progress % should slowly increase
# → ✅ Currently 1.0%
```

---

## 🐛 If Something Goes Wrong

### Training Stops Unexpectedly:
```bash
# Check console for Python errors
# Look for "Training failed:" messages in UI
# Check if thread is still alive
```

### Metrics Stop Updating:
```bash
# API might be caching
# Try refreshing the browser
# Check console logs
```

### Loss Goes to NaN:
```bash
# Training instability detected
# This would appear in console as error
# System would likely auto-recover to CPU
```

---

## ✅ Your System Status: **PERFECT**

**What You Should Do:**
1. ✅ Let it train (you're doing this!)
2. ✅ Check progress every 5-10 minutes
3. ✅ Wait for first episode to complete
4. ✅ Watch for console "Episode 10" messages

**Don't Worry About:**
- Episode staying at 0 (normal!)
- Rewards at 0.0 (expected early on)
- High loss values (will decrease)
- No console prints yet (need Episode 10+)

---

## 📊 Expected Timeline

| Time | Expected Behavior |
|------|-------------------|
| **0-5 min** | Setup, data loading, initial updates |
| **5-30 min** | First episode completes, Episode=1 |
| **30-60 min** | Episode 10+ prints appear, rewards improve |
| **1-3 hours** | Model learning, losses decreasing |
| **3+ hours** | Should see positive rewards, model improving |

**You're in the 5-30 minute phase!** Everything is working perfectly. 🎉

