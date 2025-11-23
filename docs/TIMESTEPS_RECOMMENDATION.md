# Total Timesteps Recommendation

**Date**: 2025-11-22  
**Data Available**: ~5.1 million bars from NT8 export

---

## Data Analysis

### Available Data
- **Total Files**: 61 NT8 export files
- **Average Bars per File**: ~84,063 bars
- **Total Estimated Bars**: **~5,127,867 bars**
- **Current Config**: 1,000,000 timesteps

### Data Breakdown
- Each bar = 1 timestep (for 1-minute timeframe)
- With `max_episode_steps: 10,000`, each episode uses 10,000 bars
- **Episodes per full dataset**: ~513 episodes

---

## Timesteps Recommendations

### Option 1: Minimum Training (1 Full Pass) ⚠️
**Total Timesteps**: **5,000,000 - 5,500,000**

**Pros:**
- Uses all available data once
- Ensures model sees entire dataset
- Good for initial training

**Cons:**
- May not be enough for complex strategies
- Model may need multiple passes to learn

**Use When:**
- Initial training run
- Testing configuration
- Limited training time

---

### Option 2: Standard Training (3-5 Passes) ✅ **RECOMMENDED**
**Total Timesteps**: **15,000,000 - 25,000,000**

**Pros:**
- Multiple passes through data (better learning)
- Model sees patterns multiple times
- Good balance of training time and quality
- Standard for RL training

**Cons:**
- Longer training time (days/weeks)
- Requires patience

**Use When:**
- Production training
- Want best model quality
- Have time for extended training

**Recommended Value**: **20,000,000 timesteps**

---

### Option 3: Extended Training (10+ Passes) 🚀
**Total Timesteps**: **50,000,000+**

**Pros:**
- Maximum learning from data
- Best for complex strategies
- Model sees patterns many times

**Cons:**
- Very long training time (weeks/months)
- Diminishing returns after 5-10 passes
- May overfit to training data

**Use When:**
- Research/experimentation
- Have unlimited time
- Need absolute best performance

---

## Current Configuration Analysis

**Current**: `total_timesteps: 1,000,000`

**Analysis:**
- Only uses **~19.5%** of available data (1M / 5.1M)
- Less than 1 full pass through data
- **Too small** for your data size

**Recommendation**: Increase to at least **5,000,000** (1 full pass) or **20,000,000** (4 passes)

---

## Practical Considerations

### Training Time Estimates

Assuming:
- **GPU**: Modern GPU (RTX 3090/4090 or similar)
- **Batch Size**: 128 (base) with Turbo mode
- **Update Frequency**: Every 4,096 steps

**Time Estimates:**

| Timesteps | Episodes | Estimated Time | Updates |
|-----------|----------|----------------|---------|
| 1,000,000 | ~100 | 2-4 hours | ~244 |
| 5,000,000 | ~500 | 10-20 hours | ~1,220 |
| 20,000,000 | ~2,000 | 2-4 days | ~4,882 |
| 50,000,000 | ~5,000 | 5-10 days | ~12,207 |

*Note: Times vary based on GPU, Turbo mode, and system performance*

---

## Recommendation for Your Situation

### Immediate Action (Fix Saturation First)
**Before increasing timesteps**, fix the model saturation issue:
1. Fix quality filters to allow initial trades
2. Increase entropy coefficient
3. Add early saturation detection

### After Fixes Applied

**Recommended Configuration:**
```yaml
training:
  total_timesteps: 20000000  # 20 million (4 passes through data)
  save_freq: 100000  # Save every 100k steps (more frequent checkpoints)
  eval_freq: 50000  # Evaluate every 50k steps
```

**Why 20 Million?**
- ✅ 4 full passes through your data
- ✅ Good balance of training time and quality
- ✅ Standard for production RL training
- ✅ Allows model to learn patterns deeply
- ✅ Reasonable training time (2-4 days)

---

## Configuration Update

### Update `configs/train_config_adaptive.yaml`:

```yaml
training:
  total_timesteps: 20000000  # Increased from 1,000,000
  save_freq: 100000  # More frequent saves for large training runs
  eval_freq: 50000  # More frequent evaluations
```

### Alternative: Progressive Training

If 20M is too long, use progressive approach:

1. **Phase 1**: 5,000,000 timesteps (1 pass) - Initial learning
2. **Evaluate**: Check if saturation fixed, performance improving
3. **Phase 2**: Continue to 20,000,000 if Phase 1 successful

---

## Summary

| Metric | Current | Recommended | Extended |
|--------|---------|-------------|----------|
| **Total Timesteps** | 1,000,000 | **20,000,000** | 50,000,000 |
| **Data Coverage** | 19.5% | **390%** (4 passes) | 975% (10 passes) |
| **Training Time** | 2-4 hours | **2-4 days** | 5-10 days |
| **Episodes** | ~100 | **~2,000** | ~5,000 |

**Final Recommendation**: **20,000,000 timesteps** (20 million)

This provides:
- ✅ 4 full passes through your 5.1M bars of data
- ✅ Sufficient training for complex strategies
- ✅ Reasonable training time
- ✅ Standard practice for RL training

