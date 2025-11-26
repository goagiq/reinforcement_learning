# Adaptive Learning + Enforcement Logic: How They Work Together

## Overview

The system has **two separate but related mechanisms** for managing Risk/Reward (R:R) ratios:

1. **Adaptive Learning** - Adjusts the target R:R threshold based on performance
2. **Enforcement Logic** - Rejects trades if actual R:R is too poor

Both mechanisms work together to improve trading performance.

---

## 1. Adaptive Learning (Target R:R)

### What It Does
Adaptive learning **adjusts the target R:R threshold** (`min_risk_reward_ratio`) based on actual trading performance.

### How It Works
- **Evaluates every 5,000 timesteps**
- **Monitors actual R:R**: `avg_win / avg_loss`
- **Adjusts target threshold**:
  - If actual R:R < 1.5: **Tightens** (increases threshold toward 2.5)
  - If actual R:R >= 2.0: **Relaxes** (decreases threshold toward 1.3)

### Floor & Ceiling (Configurable)
```yaml
# In AdaptiveConfig
min_rr_threshold: 1.3   # Floor - adaptive learning won't go below this
max_rr_threshold: 2.5   # Ceiling - adaptive learning won't go above this
```

### Where It's Used
- Saved to: `logs/adaptive_training/current_reward_config.json`
- Used by: Reward function (penalizes poor R:R)
- Used by: Environment quality filters (encourages better trades)

### Example
```
Initial: min_risk_reward_ratio = 1.2 (from config)
After 5K steps: Actual R:R = 0.73 (poor)
Action: Increase to 1.3 (tightening)
After 10K steps: Actual R:R = 1.8 (good)
Action: Decrease to 1.25 (relaxing slightly)
```

---

## 2. Enforcement Logic (Trade Rejection Floor)

### What It Does
Enforcement logic **rejects trades** if the actual R:R is catastrophically poor, preventing further losses.

### How It Works
- **Checks after 20+ trades** (needs reliable estimate)
- **Calculates actual R:R**: `avg_win / avg_loss` from recent trades
- **Rejects trade if**: `actual_rr < enforcement_floor`

### Floor (Configurable)
```python
# In AdaptiveConfig
min_rr_floor: 0.7  # Absolute minimum - reject trades if actual R:R < 0.7
```

### Where It's Used
- Saved to: `logs/adaptive_training/current_reward_config.json`
- Used by: Trading environment (rejects trades before execution)
- **Purpose**: Prevents catastrophic losses while allowing learning

### Example
```
After 20 trades: Actual R:R = 0.65 (catastrophic)
Action: Reject all trades until R:R improves
After 30 trades: Actual R:R = 0.75 (still poor, but above floor)
Action: Allow trades (reward function will still penalize poor R:R)
```

---

## 3. How They Work Together

### The Two Floors Explained

| Floor Type | Value | Purpose | When Applied |
|------------|-------|---------|--------------|
| **Adaptive Floor** | 1.3 | Prevents adaptive learning from setting target too low | During adaptive adjustments |
| **Enforcement Floor** | 0.7 | Prevents catastrophic trades from executing | Before each trade |

### Flow Diagram

```
1. Adaptive Learning (Every 5K timesteps)
   └─> Adjusts min_risk_reward_ratio (target: 1.2-2.5)
       └─> Saves to current_reward_config.json

2. Trading Environment (Every Episode Reset)
   └─> Reads min_risk_reward_ratio from config
       └─> Uses for reward function & quality filters

3. Enforcement Logic (Before Each Trade, after 20+ trades)
   └─> Calculates actual R:R from recent trades
       └─> If actual_rr < min_rr_floor (0.7): Reject trade
       └─> Otherwise: Allow trade (reward function will still penalize)
```

### Why Two Different Floors?

1. **Adaptive Floor (1.3)**:
   - **Purpose**: Ensures adaptive learning maintains a reasonable target
   - **Context**: "What should we aim for?"
   - **Higher value** because it's a target to strive toward

2. **Enforcement Floor (0.7)**:
   - **Purpose**: Prevents catastrophic trades from executing
   - **Context**: "Is this trade too dangerous to allow?"
   - **Lower value** because it allows learning while preventing disasters

### Example Scenario

```
Current State:
- Adaptive target (min_risk_reward_ratio): 1.2 (from adaptive learning)
- Actual R:R (recent trades): 0.73
- Enforcement floor: 0.7

What Happens:
1. Enforcement: 0.73 > 0.7 → Trade ALLOWED (above floor)
2. Reward Function: Penalizes poor R:R (0.73 < 1.2 target)
3. Adaptive Learning: Will increase target next evaluation (0.73 < 1.5)

Result:
- Trade executes (learning can happen)
- Reward penalized (encourages improvement)
- Target increases (tighter requirements next time)
```

---

## 4. Configuration

### Setting the Floors

**Adaptive Config** (`src/adaptive_trainer.py`):
```python
class AdaptiveConfig:
    # Adaptive target floor (what adaptive learning won't go below)
    min_rr_threshold: float = 1.3  # Floor for target
    
    # Enforcement floor (absolute minimum to allow trades)
    min_rr_floor: float = 0.7  # Floor for trade rejection
```

**Config File** (future enhancement):
```yaml
training:
  adaptive_training:
    min_rr_threshold: 1.3  # Adaptive target floor
    min_rr_floor: 0.7      # Enforcement floor
```

### Recommended Values

| Floor | Recommended | Min | Max | Reason |
|-------|-------------|-----|-----|--------|
| **Adaptive Floor** | 1.2-1.3 | 1.0 | 1.5 | Should be profitable, but not too strict |
| **Enforcement Floor** | 0.7-0.8 | 0.5 | 0.9 | Allows learning, prevents disasters |

---

## 5. Benefits

### Why This Design Works

1. **Allows Learning**: Enforcement floor (0.7) is lower than adaptive target (1.2), so agent can learn
2. **Prevents Disasters**: Enforcement floor prevents catastrophic trades (R:R < 0.7)
3. **Maintains Standards**: Adaptive floor ensures target never goes too low (never < 1.3)
4. **Gradual Improvement**: System can improve from 0.7 → 1.2 → 1.5+ over time

### Current Performance vs. Targets

```
Current: Actual R:R = 0.73
├─ Above enforcement floor (0.7) ✅ → Trades allowed
├─ Below adaptive target (1.2) ❌ → Reward penalized
└─ Below adaptive floor (1.3) ❌ → Target will increase

Expected Progression:
0.73 → 0.8 → 1.0 → 1.2 → 1.5+
```

---

## 6. Summary

- **Adaptive Learning**: Sets the **target** R:R (1.2-2.5, floor: 1.3)
- **Enforcement Logic**: Sets the **minimum acceptable** R:R (floor: 0.7)
- **Both work together**: 
  - Enforcement allows learning (0.7 floor)
  - Adaptive encourages improvement (1.2+ target)
  - Adaptive floor prevents target from going too low (1.3 minimum)

This design allows the agent to learn while preventing catastrophic losses!

