# Fix: RL-Only Trades Now Apply Quality Filters

## Problem Identified

RL-only trades (no swarm, no reasoning) were NOT applying quality filters because:
1. `quality_score` was `None` for RL-only trades
2. `should_execute()` only checked quality score if it existed: `if decision.quality_score:`
3. Result: All RL-only trades passed through if confidence >= 0.5

## Fix Applied

### 1. Calculate Quality Score for RL-Only Trades

Modified `DecisionGate.make_decision()` RL-only path (lines 166-178) to:
- Calculate `quality_score` even for RL-only trades
- Use quality scorer with RL confidence and default assumptions
- Include `quality_score` in DecisionResult

### 2. Enforce Quality Score Check

Modified `DecisionGate.should_execute()` to:
- Reject trades if `quality_scorer_enabled` but `quality_score is None`
- This ensures RL-only trades MUST calculate quality score
- Prevents trades from bypassing quality filters

## Expected Behavior

After fix:
- ✅ RL-only trades will calculate quality score
- ✅ Quality filters will be applied (0.4 confidence, 0.65 quality score, positive EV)
- ✅ Bad trades will be rejected
- ✅ Win rate should improve

## Files Modified

1. **`src/decision_gate.py`**:
   - Lines 166-212: Added quality score calculation for RL-only trades
   - Lines 769-773: Enforced quality score check (reject if None)

