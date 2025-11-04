# Continuous Learning & Continuous Training Guide

## Overview

This project supports two complementary approaches to keep models improving over time:

1. **Continuous Training**: Automatically retrain models when new data is available
2. **Continuous Learning**: Learn from live trading experiences and improve incrementally

## 🔑 Key Design Decisions

### Model Naming & Versioning

**Continuous Learning creates new versioned models** with each retraining cycle:

- **Initial save**: `models/continuous_learning_1.pt`, `continuous_learning_2.pt`, etc.
- **Versioned models**: `models/v1_20250115_143022.pt`, `v2_20250115_153045.pt`, etc.
- **Production model**: Current best version is copied to `models/best_model.pt`

**Important**: Currently, **models are NOT instrument-specific**. The same model learns from all experiences regardless of instrument (ES, MES, etc.). If you trade multiple instruments, experiences from all instruments are mixed in the buffer.

### Weight Transfer

**✅ YES - Learned weights ARE transferred** during continuous learning:

1. **Loads current production model**: `agent.load(prod_version.model_path)` (line 240 in `automated_learning.py`)
2. **Updates with new experiences**: Same agent instance is updated incrementally
3. **Saves new version**: Updated weights are saved as a new version
4. **Deploys if better**: New version becomes production if >10% Sharpe improvement

This means:
- Each retraining **starts from the previous model's weights**
- Learning is **incremental and cumulative**
- No knowledge is lost between retraining cycles

### Future Enhancement: Instrument-Specific Models

If you need instrument-specific models (e.g., separate model for ES vs MES), you would need to:
1. Create separate experience buffers per instrument
2. Modify model naming to include instrument: `best_model_ES.pt`, `best_model_MES.pt`
3. Maintain separate version managers per instrument

This is not currently implemented but could be added if needed.

---

## 🎯 What's the Difference?

### Continuous Training
- **Trigger**: New data files detected in NT8 export directory
- **Process**: Full retraining from scratch (or from checkpoint) with new data
- **When**: Periodically, when new historical data becomes available
- **Use Case**: Adapt to new market conditions, incorporate new data patterns

### Continuous Learning
- **Trigger**: Accumulated trading experiences reach threshold
- **Process**: Incremental learning from real trading experiences
- **When**: Continuously, as you trade and collect experiences
- **Use Case**: Learn from actual trading outcomes, refine strategy based on results

---

## 🔄 Continuous Training Setup

### How It Works

1. **File Monitoring**: System watches NT8 export directory for new CSV/TXT files
2. **Detection**: When new files are detected, a retraining trigger is queued
3. **Debouncing**: Waits 30 seconds after last file change to avoid multiple triggers
4. **Training**: Retrains model with new data (doesn't interrupt existing training)

### Configuration

#### Step 1: Enable Auto-Retrain Monitoring

Edit `settings.json` in project root:

```json
{
  "nt8_data_path": "C:\\Users\\YourName\\Documents\\NinjaTrader 8\\export",
  "auto_retrain_enabled": true,
  "performance_mode": "quiet"
}
```

**Note**: The auto-retrain monitor starts automatically when the API server starts (if `auto_retrain_enabled` is `true`).

#### Step 2: Configure Retraining (Optional)

Edit `configs/train_config.yaml`:

```yaml
# Auto-retraining when new data detected
auto_retrain:
  enabled: true
  queue_retrain_on_completion: true  # Queue retrain after current training completes
  min_new_data_files: 1              # Minimum files to trigger retrain
  debounce_seconds: 30                # Wait time after last file change
```

### How to Use

1. **Export data from NT8**: Export your latest trading data to the configured export directory
2. **Automatic detection**: System detects new files automatically (waits 30 seconds after last file change)
3. **Automatic training**: Training starts automatically with the latest checkpoint
4. **Notification**: You'll see a notification in the UI when training is triggered

**Note**: The system will:
- Wait 30 seconds after the last file change (debounce) before triggering
- Automatically find and resume from the latest checkpoint
- Skip if training is already running (won't interrupt active training)

### Check Monitor Status

Verify the monitor is running:

```bash
# Via API (recommended)
curl http://localhost:8200/api/settings/auto-retrain-status

# Or check backend logs when starting the server
# Look for: "✅ Auto-retrain monitoring started on: ..."
```

### Manual Trigger (If Needed)

If auto-retrain doesn't trigger automatically, you can manually start training:

```bash
# Via API
curl -X POST http://localhost:8200/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"device": "cuda", "checkpoint_path": "models/best_model.pt"}'

# Or use the UI: Go to Training tab and click "Resume Training"
```

### Troubleshooting Auto-Retrain

If new files aren't triggering training:

1. **Check monitor is running**:
   ```bash
   curl http://localhost:8200/api/settings/auto-retrain-status
   ```

2. **Check backend logs**: Look for "📁 New data detected" messages

3. **Verify file detection**: The monitor waits 30 seconds after the last file change before triggering

4. **Check training status**: If training is already running, new files will be queued (not triggered immediately)

5. **Restart API server**: If monitor isn't running, restart `start_ui.py`

---

## 🧠 Continuous Learning Setup

### How It Works

1. **Experience Collection**: During live trading, experiences are collected (state, action, reward, outcome)
2. **Annotation**: Experiences are annotated with reasoning engine insights (what worked, what didn't)
3. **Buffering**: Experiences are stored in a buffer (`data/experience_buffer/`)
4. **Retraining Trigger**: When threshold reached (e.g., 1000 new experiences), model retrains
5. **Evaluation**: New model is evaluated against current production model
6. **Deployment**: If new model is significantly better (>10% Sharpe improvement), it's deployed

### Configuration

#### Step 1: Enable Continuous Learning in Config

Edit `configs/train_config.yaml`:

```yaml
# Continuous Learning Configuration
continuous_learning:
  retrain_frequency: 1000            # Retrain every N new experiences
  min_experiences: 500                # Minimum experiences before retraining
  evaluation_episodes: 10             # Episodes for model evaluation
  min_annotated_for_finetune: 100     # Minimum annotated experiences for DeepSeek fine-tuning
  experience_buffer_size: 10000       # Maximum experiences to store
  experience_storage: "data/experience_buffer"
```

**Key Parameters:**
- `retrain_frequency`: How often to retrain (every N experiences)
- `min_experiences`: Don't retrain until you have at least this many
- `evaluation_episodes`: How many episodes to run for evaluation
- `experience_buffer_size`: Maximum experiences to keep in memory

#### Step 2: Enable During Live Trading

The continuous learning pipeline integrates with live trading. When you run live trading:

```python
from src.live_trading import LiveTradingSystem
from src.automated_learning import AutomatedLearningOrchestrator

# Load config
config = load_config("configs/train_config.yaml")

# Initialize orchestrator
orchestrator = AutomatedLearningOrchestrator(config)

# Start live trading (automatically collects experiences)
trading_system = LiveTradingSystem(config, model_path="models/best_model.pt")
trading_system.start()

# Periodically check for retraining
while trading_system.is_running():
    orchestrator.check_and_retrain(trading_system.agent)
    time.sleep(300)  # Check every 5 minutes
```

### Experience Buffer Structure

Experiences are stored with:
- **State**: Market conditions at decision time
- **Action**: Position size chosen by agent
- **Reward**: Immediate reward from environment
- **Outcome**: Post-trade result (PnL, duration, etc.)
- **Reflection**: Reasoning engine's analysis (what worked/wrong)

Example experience:
```python
{
  "timestamp": "2025-01-15T10:30:00",
  "state": [...200 features...],
  "action": 0.75,  # 75% position size
  "reward": 0.05,  # Small positive reward
  "next_state": [...],
  "done": False,
  "market_conditions": {"volatility": 0.15, "trend": "bullish"},
  "rl_confidence": 0.82,
  "reasoning_confidence": 0.78,
  "trade_outcome": {
    "pnl": 125.50,
    "duration": 1800,  # seconds
    "action": "BUY"
  },
  "reflection_insight": {
    "reasoning": "Trade succeeded because entry timing aligned with momentum",
    "lessons_learned": ["Good entry timing", "Proper position sizing"]
  }
}
```

### Automatic Workflow

```
Live Trading → Collect Experiences → Store in Buffer
                                            ↓
                    Accumulate 1000 new experiences?
                                            ↓
                    Yes → Retrain Model with New Experiences
                                            ↓
                    Evaluate New Model vs Current Production
                                            ↓
                    Improvement > 10% Sharpe?
                                            ↓
                    Yes → Deploy New Model → Continue Trading
```

---

## 🔧 Running Continuous Learning

### Option 1: Automated (Recommended)

Run the orchestrator as a background service:

```bash
# Run maintenance mode (saves buffer, prints stats)
python src/automated_learning.py --mode maintenance

# Check if retraining needed and trigger if so
python src/automated_learning.py --mode retrain

# Check if DeepSeek fine-tuning needed
python src/automated_learning.py --mode finetune

# Run all (maintenance + retrain + finetune)
python src/automated_learning.py --mode all
```

### Option 2: Integration with Live Trading

Integrate into your live trading script:

```python
from src.automated_learning import AutomatedLearningOrchestrator
import time

config = load_config("configs/train_config.yaml")
orchestrator = AutomatedLearningOrchestrator(config)

# During live trading loop
while trading:
    # ... trading logic ...
    
    # Check every 5 minutes
    if time.time() % 300 == 0:
        orchestrator.check_and_retrain(agent)
        orchestrator.trigger_deepseek_finetuning()
```

---

## 📊 Monitoring & Maintenance

### Check Experience Buffer Status

```bash
python src/automated_learning.py --mode maintenance
```

Output:
```
📊 Experience Buffer:
   Total experiences: 2,543
   Annotated: 1,892
   Buffer size: 2,543

📦 Model Versions:
   Production: v1.2.3
   Total versions: 8
```

### View Experience Statistics

The experience buffer automatically saves to:
- `data/experience_buffer/experience_buffer_YYYYMMDD.pkl`

You can load and inspect:

```python
from src.continuous_learning import ExperienceBuffer

buffer = ExperienceBuffer()
buffer.load("data/experience_buffer/experience_buffer_20250115.pkl")

print(f"Total: {buffer.total_experiences}")
print(f"Annotated: {buffer.annotated_count}")

# Get high-value experiences
high_value = buffer.get_high_value_experiences(n=10)
print(f"Best trades: {len(high_value)}")

# Get failed experiences
failed = buffer.get_failed_experiences(n=10)
print(f"Failed trades: {len(failed)}")
```

---

## 🎓 DeepSeek Fine-Tuning

When enough annotated experiences accumulate, you can fine-tune the reasoning engine:

### Automatic Generation

```bash
python src/automated_learning.py --mode finetune
```

This:
1. Extracts high-value and failed experiences
2. Formats them as training examples
3. Saves to `data/finetuning/deepseek_training_YYYYMMDD.json`

### Manual Fine-Tuning

1. Generate training data:
   ```bash
   python src/automated_learning.py --mode finetune
   ```

2. Review training data:
   ```bash
   # Check the generated file
   cat data/finetuning/deepseek_training_20250115.json
   ```

3. Run fine-tuning (see `docs/FINETUNING_GUIDE.md` for details)

---

## ⚙️ Configuration Examples

### Conservative (Slow Learning)

```yaml
continuous_learning:
  retrain_frequency: 5000           # Retrain every 5000 experiences
  min_experiences: 2000              # Need 2000 minimum
  evaluation_episodes: 20             # More thorough evaluation
```

### Aggressive (Fast Learning)

```yaml
continuous_learning:
  retrain_frequency: 500              # Retrain every 500 experiences
  min_experiences: 200                # Lower threshold
  evaluation_episodes: 5              # Faster evaluation
```

### Balanced (Recommended)

```yaml
continuous_learning:
  retrain_frequency: 1000             # Retrain every 1000 experiences
  min_experiences: 500                # Need 500 minimum
  evaluation_episodes: 10             # Standard evaluation
```

---

## 🚨 Important Considerations

### 1. Experience Quality
- Annotated experiences (with reasoning insights) are prioritized
- Ensure reasoning engine is enabled during live trading for best results

### 2. Overfitting Risk
- Too frequent retraining can overfit to recent experiences
- Use `retrain_frequency` to balance learning speed vs stability

### 3. Resource Usage
- Retraining uses CPU/GPU resources
- Don't retrain during critical trading hours if possible

### 4. Model Deployment
- New models only deploy if significantly better (>10% Sharpe)
- Previous production models are kept for rollback
- Production model is always copied to `best_model.pt` for easy access

### 5. Storage
- Experience buffer grows over time
- Old buffer files are kept but not automatically cleaned
- Monitor `data/experience_buffer/` directory size
- Versioned models accumulate: `models/v1_*.pt`, `v2_*.pt`, etc.

### 6. Instrument Handling (Current Limitation)
- **Models are currently shared across all instruments**
- If trading ES and MES, both contribute to the same model
- Experiences from all instruments are mixed in the buffer
- If you need separate models per instrument, you'll need to modify the code or run separate instances

---

## 📝 Quick Start Checklist

- [ ] Configure `settings.json` with NT8 export path
- [ ] Enable `auto_retrain_enabled: true` in settings
- [ ] Configure `continuous_learning` section in `train_config.yaml`
- [ ] Start API server (auto-retrain monitor starts automatically)
- [ ] Begin live trading (experiences collected automatically)
- [ ] Periodically run `python src/automated_learning.py --mode maintenance`
- [ ] Monitor experience buffer growth
- [ ] Watch for retraining notifications in UI

---

## 🔍 Troubleshooting

### Auto-retrain not detecting files
- Check `settings.json` has correct `nt8_data_path`
- Verify path exists and is readable
- Check logs for file watcher errors

### Experiences not being collected
- Ensure reasoning engine is enabled
- Check live trading is configured to use continuous learning
- Verify `experience_buffer_size` is sufficient

### Retraining not triggering
- Check `retrain_frequency` threshold
- Verify `min_experiences` is met
- Run maintenance mode to see buffer status

### Model not deploying
- Check if improvement threshold (10% Sharpe) is met
- Review evaluation metrics in logs
- Manually compare models using `src/model_evaluation.py`

---

## 📚 Related Documentation

- `docs/FINETUNING_GUIDE.md` - DeepSeek fine-tuning details
- `docs/AUTO_RETRAIN_QUICKSTART.md` - Quick setup guide
- `docs/TRAINING_CONFIGURATION.md` - Training config details
- `src/continuous_learning.py` - Source code with detailed comments
- `src/automated_learning.py` - Orchestrator implementation

---

## 💡 Best Practices

1. **Start Conservative**: Begin with higher `retrain_frequency` (2000-5000)
2. **Monitor Closely**: Watch first few retraining cycles carefully
3. **Keep Production Stable**: Don't retrain too frequently during active trading
4. **Validate New Models**: Always backtest new models before deploying
5. **Maintain Experience Quality**: Ensure reasoning engine provides good annotations
6. **Regular Maintenance**: Run maintenance mode weekly to clean up and review

---

**Last Updated**: 2025-01-15

