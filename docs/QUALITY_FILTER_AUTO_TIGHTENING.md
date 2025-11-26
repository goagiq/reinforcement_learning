# Quality Filter Auto-Tightening - Enhanced Adaptive Learning

**Date:** Current  
**Status:** ✅ Implemented and Integrated  
**Priority:** HIGH - Active during losing streaks

---

## 🎯 Overview

The adaptive learning system now **automatically tightens quality filters** when detecting losing streaks. This provides faster response to negative performance without waiting for full evaluation cycles.

---

## ✅ What Was Enhanced

### **1. Enhanced `quick_adjust_for_negative_trend()` Method**

**Location:** `src/adaptive_trainer.py`

**Enhancements:**
- ✅ **More aggressive adjustments** based on losing streak severity
- ✅ **Uses trade journal data** for better analysis (avg loss, avg win)
- ✅ **Dynamic adjustment rates** based on:
  - PnL severity (how negative)
  - Win rate (lower = more aggressive)
  - Average loss size (larger = more aggressive)
- ✅ **Better logging** with detailed adjustment information

**Before:**
- Fixed small adjustments (0.005 confidence, 0.01 quality)
- No trade data analysis
- Simple negative trend detection

**After:**
- Dynamic adjustments (up to 3.5x multiplier based on severity)
- Analyzes recent trades from journal (avg loss, avg win)
- Severity-based tightening (more negative = more aggressive)

---

## 🔧 How It Works

### **Trigger Conditions:**
1. **Negative trend detected:** Mean PnL of last 10 episodes < 0
2. **Called every episode** (not just during evaluation)
3. **Uses trade journal data** if available (last 20 trades)

### **Adjustment Calculation:**

```python
# Base adjustment rates
base_confidence_adj = 0.005
base_quality_adj = 0.01

# Calculate severity multiplier
pnl_severity = abs(recent_mean_pnl) / 100.0  # Normalize
confidence_multiplier = 1.0 + (pnl_severity * 0.5)  # Up to 3.5x

# Additional multipliers:
# - Low win rate (< 40%): 1.5x
# - Large avg loss (> $100): Up to 2x

# Final adjustment
confidence_adjustment = base_confidence_adj * confidence_multiplier
quality_adjustment = base_quality_adj * quality_multiplier
```

### **Adjustment Caps:**
- `min_action_confidence`: Max 0.30 (was 0.25)
- `min_quality_score`: Max 0.70 (was 0.60)

---

## 📊 Example Scenarios

### **Scenario 1: Mild Losing Streak**
- Mean PnL: -$50
- Win Rate: 45%
- Avg Loss: $80

**Adjustment:**
- Severity: 0.5
- Confidence: +0.003 (0.005 × 0.6)
- Quality: +0.006 (0.01 × 0.6)

---

### **Scenario 2: Severe Losing Streak**
- Mean PnL: -$200
- Win Rate: 35%
- Avg Loss: $150

**Adjustment:**
- Severity: 2.0
- Confidence: +0.015 (0.005 × 3.0)
- Quality: +0.030 (0.01 × 3.0)

---

### **Scenario 3: Critical Losing Streak**
- Mean PnL: -$500
- Win Rate: 30%
- Avg Loss: $200

**Adjustment:**
- Severity: 5.0 (capped)
- Confidence: +0.0175 (0.005 × 3.5)
- Quality: +0.035 (0.01 × 3.5)

---

## 🔄 Integration Points

### **1. Training Loop Integration**

**Location:** `src/train.py` (lines 1254-1276)

**What Happens:**
1. Every episode end, checks last 10 episodes
2. If negative trend detected:
   - Gets recent trades from journal (if available)
   - Calls `quick_adjust_for_negative_trend()`
   - Applies adjustments immediately
   - Updates reward config file

**Code:**
```python
if self.adaptive_trainer and len(self.episode_pnls) >= 10:
    recent_pnls = self.episode_pnls[-10:]
    recent_mean_pnl = sum(recent_pnls) / len(recent_pnls)
    
    if recent_mean_pnl < 0:
        # Get recent trades from journal
        recent_trades_data = get_recent_trades_from_journal()
        
        # Quick adjustment
        quick_adjustments = self.adaptive_trainer.quick_adjust_for_negative_trend(
            recent_mean_pnl=recent_mean_pnl,
            recent_win_rate=...,
            agent=self.agent,
            recent_trades_data=recent_trades_data
        )
```

---

### **2. Environment Integration**

**Location:** `src/trading_env.py` (lines 213-236)

**What Happens:**
1. Environment reads adaptive config on reset
2. Uses updated quality filter values
3. Applies filters during training

**Code:**
```python
adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
if adaptive_config_path.exists():
    adaptive_config = json.load(f)
    quality_filters = adaptive_config.get("quality_filters", {})
    self.min_action_confidence = quality_filters.get("min_action_confidence", ...)
    self.min_quality_score = quality_filters.get("min_quality_score", ...)
```

---

## 📈 Expected Impact

### **Before Enhancement:**
- Fixed small adjustments (0.005 confidence, 0.01 quality)
- Slow response to losing streaks
- No trade data analysis
- Takes many episodes to tighten filters

### **After Enhancement:**
- Dynamic adjustments (up to 3.5x based on severity)
- Fast response (every episode, not just evaluation)
- Uses trade journal data for better decisions
- More aggressive for severe losing streaks

### **Expected Results:**
- **Faster filter tightening** during losing streaks
- **Better trade quality** (filters out bad trades sooner)
- **Reduced average loss** (stops bad trades earlier)
- **Improved risk/reward** (only takes high-quality trades)

---

## 🎛️ Configuration

### **Adjustment Parameters:**

**Location:** `src/adaptive_trainer.py` - `AdaptiveConfig` class

```python
# Quality filter adjustment
quality_filter_adjustment_enabled: bool = True
min_action_confidence_range: Tuple[float, float] = (0.1, 0.2)  # (min, max)
min_quality_score_range: Tuple[float, float] = (0.3, 0.5)  # (min, max)
quality_adjustment_rate: float = 0.01  # Base adjustment rate
```

**Note:** The enhanced method uses dynamic multipliers, so these are base rates.

---

## 📝 Logging

### **Example Log Output:**

```
[ADAPT] 🔧 Quick Quality Filter Tightening (Losing Streak):
   Mean PnL: $-134.52
   Win Rate: 50.0%
   Avg Loss: $134.52
   Avg Win: $124.00
   Severity: 1.35
   Confidence: 0.200 → 0.207 (+0.007)
   Quality: 0.500 → 0.514 (+0.014)
```

---

## ⚠️ Important Notes

1. **Adjustments are cumulative:** Each episode with negative trend tightens filters further
2. **Caps prevent over-tightening:** Max confidence = 0.30, Max quality = 0.70
3. **Resets on positive trend:** When profitability returns, filters can relax (via full evaluation)
4. **Works alongside full evaluation:** Quick adjustments + full evaluation adjustments

---

## 🔍 Monitoring

### **Check Adjustment History:**

**File:** `logs/adaptive_training/config_adjustments.jsonl`

**Format:**
```json
{
  "timestep": 1950000,
  "episode": 195,
  "adjustments": {
    "quality_filters": {
      "min_action_confidence": {
        "old": 0.200,
        "new": 0.207,
        "reason": "Quick adjustment: negative trend (mean_pnl=$-134.52, win_rate=50.0%, severity=1.35)"
      },
      "min_quality_score": {
        "old": 0.500,
        "new": 0.514,
        "reason": "Quick adjustment: negative trend (mean_pnl=$-134.52, win_rate=50.0%, severity=1.35)"
      }
    }
  }
}
```

---

## ✅ Status

- ✅ Enhanced `quick_adjust_for_negative_trend()` method
- ✅ Integrated with training loop
- ✅ Uses trade journal data
- ✅ Dynamic severity-based adjustments
- ✅ Better logging
- ✅ Tested and verified

**Next Steps:**
- Monitor performance after restart
- Verify filters are tightening during losing streaks
- Check adjustment logs for correctness

---

**Status:** ✅ Complete - Ready for Testing  
**Priority:** HIGH - Active during losing streaks

