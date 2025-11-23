# Implementation Plan: Adaptive Learning Fixes

**Priority**: CRITICAL  
**Goal**: Fix adaptive learning to detect and respond to unprofitability aggressively

---

## Summary of Issues Found

1. ✅ Adaptive learning WAS running (822 snapshots)
2. ❌ Did NOT detect 0% win rate during training
3. ❌ Adjustments too conservative (0.01 increments)
4. ❌ No pause/resume logic for critical failures
5. ❌ No reward for good performance
6. ❓ DecisionGate may not read adaptive parameters

---

## Implementation Steps

### Step 1: Update Adaptive Trainer (CRITICAL)

**File**: `src/adaptive_trainer.py`

**Changes Needed**:

1. **Add tracking variables** (in `__init__`):
```python
self.consecutive_low_win_rate = 0
self.consecutive_good_performance = 0
self.training_paused = False
self.pause_reason = None
```

2. **Improve profitability detection** (in `_analyze_and_adjust`):
```python
# Replace current weak profitability check with aggressive version
if not profitability_check["is_profitable"]:
    self.consecutive_low_win_rate += 1
    
    print(f"[CRITICAL] UNPROFITABLE: Win rate {profitability_check['current_win_rate']:.1%} < breakeven {profitability_check['breakeven_win_rate']:.1%}")
    print(f"   Consecutive unprofitable evaluations: {self.consecutive_low_win_rate}")
    
    # AGGRESSIVE tightening when unprofitable
    if self.consecutive_low_win_rate >= 2:
        old_confidence = self.current_min_action_confidence
        old_quality = self.current_min_quality_score
        old_rr = self.current_min_risk_reward_ratio
        
        # Aggressive increases (5x normal rate)
        self.current_min_action_confidence = min(0.25,
            self.current_min_action_confidence + 0.05)
        self.current_min_quality_score = min(0.60,
            self.current_min_quality_score + 0.10)
        self.current_min_risk_reward_ratio = min(3.0,
            self.current_min_risk_reward_ratio + 0.5)
        
        adjustments["quality_filters"] = {
            "min_action_confidence": {"old": old_confidence, "new": self.current_min_action_confidence},
            "min_quality_score": {"old": old_quality, "new": self.current_min_quality_score},
            "min_risk_reward_ratio": {"old": old_rr, "new": self.current_min_risk_reward_ratio},
            "reason": f"AGGRESSIVE: Unprofitable for {self.consecutive_low_win_rate} evaluations"
        }
        print(f"[ADAPT] AGGRESSIVE tightening: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, quality {old_quality:.3f}->{self.current_min_quality_score:.3f}, R:R {old_rr:.2f}->{self.current_min_risk_reward_ratio:.2f}")
    
    # PAUSE training if win rate < 30% for 3+ evaluations
    if snapshot.win_rate < 0.30 and self.consecutive_low_win_rate >= 3:
        self.training_paused = True
        self.pause_reason = f"Win rate {snapshot.win_rate:.1%} < 30% for {self.consecutive_low_win_rate} evaluations"
        self._save_pause_state()
        print(f"\n{'='*70}")
        print(f"[CRITICAL] TRAINING PAUSED")
        print(f"{'='*70}")
        print(f"Reason: {self.pause_reason}")
        print(f"Current win rate: {snapshot.win_rate:.1%}")
        print(f"Breakeven win rate: {profitability_check['breakeven_win_rate']:.1%}")
        print(f"Expected value: ${profitability_check['expected_value']:.2f}")
        print(f"\n[ACTION REQUIRED] Review performance and adjust parameters before resuming")
        print(f"Checkpoint saved. Resume with: --checkpoint models/checkpoint_{snapshot.timestep}.pt")
        print(f"{'='*70}\n")
else:
    # Reset counter when profitable
    if self.consecutive_low_win_rate > 0:
        print(f"[OK] Profitability restored! Resetting consecutive low win rate counter")
    self.consecutive_low_win_rate = 0
```

3. **Add reward good performance**:
```python
# After profitability check, add:
if snapshot.win_rate > 0.50:
    self.consecutive_good_performance += 1
    
    if self.consecutive_good_performance >= 2:
        # Reward good performance - relax filters slightly
        old_confidence = self.current_min_action_confidence
        old_quality = self.current_min_quality_score
        
        self.current_min_action_confidence = max(0.10,
            self.current_min_action_confidence - 0.02)
        self.current_min_quality_score = max(0.30,
            self.current_min_quality_score - 0.05)
        
        adjustments["quality_filters"] = {
            "min_action_confidence": {"old": old_confidence, "new": self.current_min_action_confidence},
            "min_quality_score": {"old": old_quality, "new": self.current_min_quality_score},
            "reason": f"Rewarding good performance: Win rate {snapshot.win_rate:.1%} > 50%"
        }
        print(f"[ADAPT] Rewarding good performance: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, quality {old_quality:.3f}->{self.current_min_quality_score:.3f}")
else:
    self.consecutive_good_performance = 0
```

4. **Add pause state methods**:
```python
def _save_pause_state(self):
    """Save pause state to file"""
    pause_file = self.log_dir / "training_paused.json"
    with open(pause_file, 'w') as f:
        json.dump({
            "paused": True,
            "reason": self.pause_reason,
            "timestamp": datetime.now().isoformat(),
            "checkpoint": f"checkpoint_{self.last_eval_timestep}.pt"
        }, f, indent=2)

def is_training_paused(self) -> bool:
    """Check if training is paused"""
    return self.training_paused
```

5. **Update check_trading_activity to penalize no trading more aggressively**:
```python
# In check_trading_activity, when no trades detected:
# Increase inaction penalty more aggressively (already implemented, but verify)
# Increase exploration more aggressively (already implemented, but verify)
```

### Step 2: Update Trainer to Check Pause State

**File**: `src/train.py`

**Changes Needed**:

In training loop, after adaptive evaluation:
```python
if self.adaptive_trainer:
    if self.adaptive_trainer.is_training_paused():
        print("\n[PAUSED] Training paused by adaptive learning system")
        print("Saving final checkpoint...")
        # Save checkpoint
        checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
        self.agent.save_with_training_state(...)
        print(f"Checkpoint saved: {checkpoint_path}")
        print("Training stopped. Review and adjust parameters before resuming.")
        break  # Exit training loop
```

### Step 3: Verify DecisionGate Integration

**File**: `src/decision_gate.py`

**Check**: Does DecisionGate read `current_reward_config.json`?

**If Not**, add in `__init__` or `should_execute()`:
```python
# Read adaptive parameters if available
adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
if adaptive_config_path.exists():
    try:
        with open(adaptive_config_path, 'r') as f:
            adaptive_config = json.load(f)
            quality_filters = adaptive_config.get("quality_filters", {})
            
            # Override quality scorer config with adaptive values
            if quality_filters:
                self.quality_scorer.min_quality_score = quality_filters.get(
                    "min_quality_score", 
                    self.quality_scorer.min_quality_score
                )
                # Note: min_action_confidence is handled in make_decision()
    except Exception as e:
        print(f"[WARN] Could not load adaptive config: {e}")
```

### Step 4: Update Configuration

**File**: `configs/train_config_adaptive.yaml`

**Changes**:
```yaml
adaptive_training:
  enabled: true
  eval_frequency: 5000  # More frequent (was 10000)
  eval_episodes: 10  # More episodes (was 3)
  min_trades_per_episode: 0.3
  min_win_rate: 0.40  # Increased from 0.35
  target_sharpe: 0.5
  
  # More aggressive adjustments
  quality_adjustment_rate: 0.05  # Increased from 0.01
  rr_adjustment_rate: 0.2  # Increased from 0.1
  
  # Pause/resume
  pause_on_critical_failure: true
  critical_win_rate_threshold: 0.30
  critical_failure_count: 3
  
  # Reward good performance
  reward_good_performance: true
  good_win_rate_threshold: 0.50
  good_performance_count: 2
```

**Also update initial quality filters**:
```yaml
environment:
  reward:
    quality_filters:
      min_action_confidence: 0.20  # Increased from 0.15
      min_quality_score: 0.50  # Increased from 0.40
    min_risk_reward_ratio: 2.0  # Increased from 1.5
```

### Step 5: Create Data Verification Script

**New File**: `verify_training_data.py`

```python
"""Verify all training data was processed"""
from pathlib import Path
import pandas as pd

def verify_training_data():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    raw_files = list(raw_dir.glob("*.csv"))
    processed_files = list(processed_dir.glob("*.csv"))
    
    print(f"Raw data files: {len(raw_files)}")
    print(f"Processed files: {len(processed_files)}")
    
    missing = []
    for raw_file in raw_files:
        processed = processed_dir / raw_file.name
        if not processed.exists():
            missing.append(raw_file.name)
            print(f"[WARN] {raw_file.name} not processed!")
    
    if not missing:
        print("[OK] All raw files were processed")
    
    # Check file sizes
    for raw_file in raw_files:
        processed = processed_dir / raw_file.name
        if processed.exists():
            raw_size = raw_file.stat().st_size
            proc_size = processed.stat().st_size
            print(f"{raw_file.name}: {raw_size:,} bytes -> {proc_size:,} bytes")

if __name__ == "__main__":
    verify_training_data()
```

### Step 6: Find and Evaluate Good Checkpoint

**Action**: Evaluate `checkpoint_7200000.pt` (closest to episode 723)

```bash
python evaluate_model_performance.py --model models/checkpoint_7200000.pt --episodes 20
```

**If performance is good** (win rate > 50%), use as starting point.

---

## Testing Plan

1. **Test Aggressive Adjustments**:
   - Simulate unprofitable scenario (win rate < 40%)
   - Verify filters tighten aggressively
   - Verify pause triggers at win rate < 30%

2. **Test Good Performance Reward**:
   - Simulate good performance (win rate > 50%)
   - Verify filters relax slightly

3. **Test DecisionGate Integration**:
   - Verify DecisionGate reads adaptive parameters
   - Verify parameters are applied

4. **Test Pause/Resume**:
   - Trigger pause condition
   - Verify checkpoint saved
   - Verify training stops

---

## Expected Results

After implementing:

1. **Adaptive Learning Will**:
   - Detect unprofitability immediately
   - Tighten filters aggressively (5x rate)
   - Pause training when win rate < 30%
   - Reward good performance

2. **Training Will**:
   - Start from better checkpoint
   - Adapt more quickly
   - Pause on critical failures
   - Converge to profitability

---

## Next Steps

1. ✅ Implement code changes (Steps 1-5)
2. ✅ Test changes
3. ✅ Find and evaluate good checkpoint
4. ✅ Verify training data
5. ✅ Restart training with improved system

---

**Status**: Ready for implementation

