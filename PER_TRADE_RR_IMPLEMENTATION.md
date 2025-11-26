# Per-Trade R:R Implementation - Immediate Feedback

## Problem

**Aggregate R:R is a lagging indicator:**
- R:R = avg_win / avg_loss (calculated from ALL trades)
- Agent doesn't see immediate connection between actions and R:R
- R:R changes slowly (needs many trades to change)
- Agent can't learn from individual trades

## Solution: Per-Trade R:R Tracking

### Implementation

1. **Track R:R for each trade at exit:**
   - Calculate R:R = (exit_price - entry_price) / (entry_price - stop_loss_price)
   - Store in `recent_trades_rr` list
   - Provides immediate feedback when trade closes

2. **Penalize trades that exit before target R:R:**
   - If trade exits at 0.5:1 R:R when target is 2.0:1 → penalty
   - Penalty: 20% of reward for exiting before target
   - Immediate negative feedback for poor trade management

3. **Reward trades that achieve good R:R:**
   - If trade exits at 2.0:1+ R:R → bonus
   - Bonus: Up to 15% for achieving target R:R
   - Immediate positive feedback for good trade management

4. **Strengthened aggregate R:R penalty:**
   - Increased from 30% to 50% maximum penalty
   - Applied when aggregate R:R < required R:R

## Code Changes

### Added Per-Trade R:R Tracking

```python
# In __init__:
self.recent_trades_rr = []  # Track R:R of each trade at exit

# At each trade exit (stop loss, position closed, position reversed):
# Calculate trade R:R
entry_price = self._last_entry_price
stop_loss_price = entry_price * (1 - position_direction * self.stop_loss_pct)
risk = abs(entry_price - stop_loss_price)
reward_distance = abs(current_price - entry_price)
trade_rr = reward_distance / risk if risk > 0 else 0.0
self.recent_trades_rr.append(trade_rr)
```

### Added Per-Trade R:R Penalty in Reward Function

```python
# Per-trade R:R penalty (IMMEDIATE feedback)
if last_trade_rr > 0 and last_trade_rr < required_rr:
    # Trade was profitable but exited too early - STRONG penalty
    per_trade_rr_penalty = 0.20 * (required_rr - last_trade_rr) / required_rr
    reward -= per_trade_rr_penalty

# Reward trades that achieve good R:R
if last_trade_rr >= required_rr:
    rr_bonus = min(0.15, (last_trade_rr - required_rr) / required_rr * 0.15)
    reward += rr_bonus
```

### Added Reward Logging

- Log reward components every 100 steps
- Shows aggregate R:R, penalties, and bonuses
- Helps verify penalties are being applied

## Expected Impact

### Immediate Benefits:
1. **Agent sees immediate feedback** when trade exits
2. **Penalizes early exits** before target R:R
3. **Rewards good trade management** (letting winners run)
4. **Stronger penalties** for poor aggregate R:R (50% max)

### Expected Learning:
- Agent should learn to hold winners longer
- Agent should learn to exit at target R:R (2.0:1)
- Agent should improve per-trade R:R over time
- Aggregate R:R should improve toward 2.0:1

## Next Steps

1. ✅ **DONE:** Per-trade R:R tracking
2. ✅ **DONE:** Per-trade R:R penalty
3. ✅ **DONE:** Reward logging
4. ⚠️ **TODO:** Monitor training to see if R:R improves
5. ⚠️ **TODO:** Adjust penalty strength if needed

