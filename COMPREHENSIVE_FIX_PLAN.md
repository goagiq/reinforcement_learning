# Comprehensive Fix Plan: Zero Metrics + Continuing Losses

## Issue 1: Mean (Last 10 Episodes) Showing Zeros

### Root Cause:
- `episode_pnls`, `episode_equities`, `episode_win_rates` are NOT saved in checkpoints
- When resuming from checkpoint, these lists start empty
- Only populated from NEW episodes after resume
- So "Mean (Last 10 Episodes)" shows zeros until 10+ episodes complete

### Fix:
1. **Save episode metrics in checkpoints** (modify `save_with_training_state`)
2. **Load episode metrics from checkpoints** (modify checkpoint loading)
3. **Fallback to database** if lists are empty (read from trading_journal.db)

## Issue 2: Losses Continuing to Climb

### Critical Findings:
- R:R: **0.71:1** (recent: **0.57:1** - getting WORSE!)
- Total P&L: **-$1,076,635** (was -$76,645)
- Agent is NOT learning to improve R:R
- Win rate: 45.33% but losing money rapidly

### Root Causes:

#### A. Reward Function May Not Be Strong Enough
- R:R penalty: 30% max
- But agent isn't learning → penalties aren't effective
- Need to verify penalties are actually applied

#### B. Agent May Not Understand R:R Connection
- R:R calculated from aggregate stats (avg_win / avg_loss)
- Agent doesn't see immediate connection between actions and R:R
- R:R is lagging indicator (needs many trades)

#### C. Stop Loss May Be Cutting Winners
- Stop loss: 2.5%
- Average win: $89.96 (should be $254 for 2.0:1 R:R)
- Suggests winners are being cut too early

#### D. Reward Signal May Have Conflicts
- Multiple reward components (PnL, R:R penalty, bonuses)
- Agent may be optimizing for wrong signal
- Need to simplify reward signal

## Immediate Actions

### Priority 1: Fix Mean Metrics Display
1. Save episode metrics in checkpoints
2. Load episode metrics from checkpoints
3. Add database fallback for historical data

### Priority 2: Strengthen R:R Learning
1. Add per-trade R:R tracking (not just aggregate)
2. Penalize trades that exit before target R:R
3. Reward trades that achieve good R:R

### Priority 3: Verify Reward Function
1. Add logging to confirm R:R penalties applied
2. Check if reward signal is strong enough
3. Verify agent is seeing penalty/reward connection

### Priority 4: Review Stop Loss Logic
1. Check if stop loss is actually enforced
2. Verify stop loss isn't cutting winners too early
3. Consider trailing stop loss instead

