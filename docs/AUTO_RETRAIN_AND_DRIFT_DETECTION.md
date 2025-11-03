# Automatic Retraining & Drift Detection

## 🎯 Quick Summary

Two powerful features now implemented:

1. **Auto-Retrain**: Automatically detects new data files and triggers retraining
2. **Drift Detection**: Monitors live trading performance and alerts when model degrades

Both are designed to **never interrupt** your current training! ✅

---

## 🔄 Automatic Retraining

### **How It Works**

1. **File Watcher** monitors your NT8 export directory
2. When new CSV/TXT files are detected → waits 30 seconds (debounce)
3. If no training running → triggers retraining
4. If training running → queues for later (doesn't interrupt)

### **Setup (One-Time)**

1. Open **Settings** in UI
2. Enter NT8 export path: `C:\Users\sovan\Documents\NinjaTrader 8\export`
3. Check **"Automatic Retraining"** checkbox
4. Click **Save**

**Done!** Monitoring starts automatically on next backend restart.

### **What Gets Monitored**

- `ES_1min.csv`, `ES_5min.csv`, `ES_15min.csv` (any timeframe)
- `MES_*.csv` files
- Any CSV/TXT file matching `[INSTRUMENT]_[TIMEFRAME]min.*`
- Ignores files that are still being written (size checking)

### **Safety Features**

✅ **Debounce**: Waits 30s after last file change (prevents duplicate triggers)  
✅ **No Interruption**: Won't start if training already running  
✅ **File Tracking**: Remembers processed files (no duplicates)  
✅ **Path Fallback**: Falls back to polling if path unavailable  

---

## 🚨 Drift Detection

### **How It Works**

1. **Monitors** live trading performance in real-time
2. **Compares** current metrics vs baseline (from training/test)
3. **Alerts** when thresholds breached
4. **Recommends** rollback to previous model if severe

### **Metrics Monitored**

| Metric | Threshold | Example Alert |
|--------|-----------|---------------|
| **Win Rate** | 10% drop | 55% → 45% = Alert |
| **Sharpe Ratio** | 0.3 drop | 1.0 → 0.7 = Alert |
| **Profit Factor** | 30% drop | 1.5 → 1.0 = Alert |
| **Consecutive Losses** | 5 in a row | Loss streak = Alert |

### **Alert Types**

- **Warning**: Performance degrading, monitor closely
- **Critical**: Severe degradation, rollback recommended

### **Configuration**

In `configs/train_config_gpu_optimized.yaml`:

```yaml
drift_detection:
  enabled: true                       # Enable/disable
  baseline_metrics:
    win_rate: 0.55                    # Your trained model's baseline
    sharpe_ratio: 1.0
    profit_factor: 1.5
    max_drawdown: 0.15
  thresholds:
    win_rate_drop: 0.10               # Customize thresholds
    sharpe_drop: 0.30
    consecutive_losses: 5
```

**Update baselines** after successful training to match your model's performance.

---

## 📊 How They Work Together

### **Normal Operation:**

```
Day 1: Train model → baseline metrics = {win: 0.55, sharpe: 1.2}
Day 2-5: Live trade with drift detection monitoring
Day 6: Add new data → auto-retrain detects files
       → Training starts automatically
       → Doesn't interrupt current session
Day 7: Training completes → new model deployed
       → Baseline metrics updated
       → Drift detection resets with new baseline
```

### **Degradation Scenario:**

```
Week 1: Model performing well (55% win rate, Sharpe 1.2)
Week 2: Drift monitor detects: 45% win rate, Sharpe 0.8
        → Alert: "Model degradation detected!"
        → Recommendation: Rollback to previous version
Week 3: You approve rollback → system reverts
        → Performance returns to baseline
```

---

## ⚙️ Configuration Guide

### **Backend (settings.json)**

```json
{
  "nt8_data_path": "C:\\Users\\sovan\\Documents\\NinjaTrader 8\\export",
  "auto_retrain_enabled": true,
  "performance_mode": "quiet"
}
```

### **Training Config (drift detection)**

```yaml
drift_detection:
  enabled: true
  baseline_metrics:
    win_rate: 0.55        # UPDATE after training!
    sharpe_ratio: 1.0     # UPDATE after training!
  thresholds:
    win_rate_drop: 0.10   # Customize to your tolerance
```

---

## 🎮 Usage Examples

### **Enable Auto-Retrain**

**Via UI:**
1. Settings → NT8 Data Path → Enter path
2. Check "Automatic Retraining"
3. Save
4. Backend auto-starts monitoring

**Via settings.json:**
```json
{
  "nt8_data_path": "C:\\Users\\sovan\\Documents\\NinjaTrader 8\\export",
  "auto_retrain_enabled": true
}
```

Then restart backend.

### **Monitor Drift**

**During live trading:**
```python
# System automatically tracks and alerts
# Check console for warnings:
# 🚨 Win rate dropped from 55% to 45% (drop: 10%)
#    Consider rolling back to previous model version
```

**Via API:**
```bash
curl http://localhost:8200/api/trading/drift-status
```

---

## 🔒 Safety Guarantees

### **Auto-Retrain Never:**
- ❌ Interrupts ongoing training
- ❌ Triggers duplicate retraining
- ❌ Processes incomplete files
- ❌ Monitors non-data files

### **Drift Detection Never:**
- ❌ Auto-rollbacks without your approval
- ❌ Alerts on insufficient data (<20 trades)
- ❌ Triggers false alerts (30s debounce)
- ❌ Loses historical data

---

## 📈 What You See

### **Auto-Retrain:**
```
Backend console:
✅ Auto-retrain monitoring started on: C:\Users\...\export
📁 New file detected: ES_1min.csv
🚀 Triggering retrain for 1 new file(s)
⚠️  Training already in progress. New data detected but will not retrain yet.
```

### **Drift Detection:**
```
Live trading console:
📊 Signal sent: buy @ size 0.75
📊 Signal sent: sell @ size -0.50
...
🚨 Win rate dropped from 55.0% to 44.0% (drop: 11.0%)
    Consider rolling back to previous model version
```

**UI Notification:**
```
[Popup] Auto-retrain triggered
New data detected: ES_1min.csv
Retraining recommended when current training completes.
[View Details] [Dismiss]
```

---

## ❓ FAQ

**Q: Will auto-retrain interrupt my current training?**  
A: **No!** It only triggers when no training is running.

**Q: How do I know drift detection is working?**  
A: Check backend logs during live trading. Warnings appear as performance degrades.

**Q: Can I customize alert thresholds?**  
A: Yes! Edit `drift_detection.thresholds` in config file.

**Q: What if path doesn't exist yet?**  
A: System polls every minute until path becomes available, then starts monitoring.

**Q: Do I need to restart backend to enable auto-retrain?**  
A: Not if you set it via UI. Monitoring starts automatically.

**Q: How do I disable auto-retrain?**  
A: Uncheck "Automatic Retraining" in Settings, or set `auto_retrain_enabled: false` in `settings.json`.

---

## 🚀 Quick Start

1. **Settings** → Set NT8 path → Check "Auto-Retrain" → Save ✅
2. **Train** your first model → note baseline metrics ✅
3. **Update** drift detection baselines in config ✅
4. **Start** live trading → drift detection monitors ✅
5. **Export** new data from NT8 → auto-retrain triggers ✅

**That's it!** System handles the rest automatically.

---

## 📚 Related Docs

- **[PERFORMANCE_MODE.md](PERFORMANCE_MODE.md)** - Dynamic training speed
- **[RESUME_TRAINING_QUICKSTART.md](RESUME_TRAINING_QUICKSTART.md)** - Checkpoints
- **[MODEL_DRIFT_AND_ROLLBACK_PROPOSAL.md](MODEL_DRIFT_AND_ROLLBACK_PROPOSAL.md)** - Full rollback system

---

**Both features work silently in the background, never interrupting your training!** 🎉

