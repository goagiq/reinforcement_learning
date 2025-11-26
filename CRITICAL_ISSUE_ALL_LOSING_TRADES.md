# CRITICAL ISSUE: All Trades Are Losing (0% Win Rate)

## Problem

After resuming training from checkpoint 1M:
- **Total Trades**: 902 in just 142 episodes (~2k timesteps)
- **Winning Trades**: 0
- **Losing Trades**: 902
- **Win Rate**: 0.0% (ALL TRADES ARE LOSING!)

## Root Cause

**DecisionGate RL-only trades are NOT applying quality filters!**

When DecisionGate processes RL-only trades (no swarm, no reasoning), it:
1. ✅ Returns action and confidence directly
2. ❌ Does NOT calculate quality_score
3. ❌ Does NOT calculate expected_value
4. ❌ Does NOT apply quality filters

Then `should_execute()` checks:
1. ✅ Confidence threshold (works if confidence < 0.5)
2. ❌ Confluence requirement (doesn't work - set to 0 for training)
3. ❌ Quality score (doesn't work - quality_score is None for RL-only!)

**Result**: All trades pass through if confidence >= 0.5!

## Fix Required

**Add quality score and expected value calculation to RL-only trades** in DecisionGate.make_decision().

The RL-only path (lines 166-178) should:
1. Calculate quality_score (even without swarm)
2. Calculate expected_value (from recent trade history)
3. Apply quality filters before approving trade

## Impact

Currently, the tightened quality filters (0.4 confidence, 0.65 quality score, positive EV) are **NOT being applied** to RL-only trades during training, which is why we're seeing 902 bad trades with 0% win rate.

