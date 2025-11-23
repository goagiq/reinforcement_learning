# Adaptive Training System

## Overview

The Adaptive Training System automatically monitors model performance during training and intelligently adjusts parameters **without stopping training**. This eliminates the need to manually stop, evaluate, and restart training.

## Key Features

### 1. **Automatic Evaluation**
- Evaluates model every N timesteps (configurable, default: 10,000)
- Runs 3 evaluation episodes to assess performance
- Tracks: trades, win rate, Sharpe ratio, returns

### 2. **Intelligent Parameter Adjustment**
Automatically adjusts:
- **Entropy Coefficient** (exploration): Increases if model isn't trading
- **Inaction Penalty**: Increases if model stays flat too often
- **Learning Rate**: Reduces if performance plateaus

### 3. **Auto-Save on Improvement**
- Automatically saves best model when performance improves by 5%+
- No need to manually check and save

### 4. **Performance Tracking**
- Maintains history of all evaluations
- Tracks trends (improving/declining/stable)
- Logs all adjustments for analysis

## How It Works

### During Training

1. **Every 10,000 timesteps** (configurable):
   - System pauses training briefly
   - Loads current model checkpoint
   - Runs 3 evaluation episodes
   - Analyzes performance metrics

2. **If No Trades Detected**:
   - Increases `entropy_coef` (encourages exploration)
   - Increases `inaction_penalty` (penalizes staying flat)
   - Adjustments are applied immediately to agent

3. **If Performance Plateaus**:
   - Reduces learning rate gradually
   - Helps model converge better

4. **If Significant Improvement**:
   - Automatically saves as `best_model.pt`
   - Logs improvement percentage

### Adaptive Reward Function

The reward function now reads adaptive penalties from:
```
logs/adaptive_training/current_reward_config.json
```

This allows the penalty to increase during training if the model isn't trading, without restarting.

## Configuration

### Enable Adaptive Training

Edit `configs/train_config_adaptive.yaml`:

```yaml
training:
  adaptive_training:
    enabled: true  # Enable adaptive system
    eval_frequency: 10000  # Evaluate every N timesteps
    eval_episodes: 3  # Episodes per evaluation
    min_trades_per_episode: 0.5  # Minimum trades expected
    auto_save_on_improvement: true
    improvement_threshold: 0.05  # 5% improvement triggers save
```

### Starting Training

```bash
uv run python src/train.py --config configs/train_config_adaptive.yaml --device cuda
```

The system will:
- ✅ Start training normally
- ✅ Automatically evaluate every 10k steps
- ✅ Adjust parameters if needed
- ✅ Save best models automatically
- ✅ Continue training without interruption

## Monitoring

### View Performance History

```bash
# View all performance snapshots
cat logs/adaptive_training/performance_snapshots.jsonl

# View parameter adjustments
cat logs/adaptive_training/config_adjustments.jsonl
```

### Check Current Status

The system prints status during training:
```
======================================================================
ADAPTIVE EVALUATION (Timestep: 100,000, Episode: 15)
======================================================================

📊 Performance Metrics:
   Total Trades: 12
   Win Rate: 45.0%
   Total Return: 2.35%
   Sharpe Ratio: 0.52

🔧 Adaptive Adjustments:
   entropy_coef: {'old': 0.05, 'new': 0.06, 'reason': 'Low trade activity'}
```

## Key Improvements Made

### 1. **Action Threshold Reduced**
- **Before**: `abs(position_change) > 0.01` (1% threshold)
- **After**: `abs(position_change) > 0.001` (0.1% threshold)
- **Impact**: Smaller actions now trigger trades

### 2. **Enhanced Reward Function**
- Adaptive inaction penalty (starts at 0.0001, can increase to 0.001)
- Stronger profit bonuses
- Reduced loss penalties (encourages learning)

### 3. **Optimized Config**
- `entropy_coef`: 0.025 → 0.05 (doubled exploration)
- `risk_penalty`: 0.1 → 0.05 (reduced risk aversion)
- `drawdown_penalty`: 0.1 → 0.05 (reduced penalty)

## Troubleshooting

### Model Still Not Trading?

1. **Check evaluation logs**:
   ```bash
   tail -f logs/adaptive_training/performance_snapshots.jsonl
   ```

2. **Verify adjustments are being applied**:
   - Look for "Adaptive Adjustments" messages during training
   - Check `config_adjustments.jsonl` for history

3. **Manually increase parameters**:
   - Edit `configs/train_config_adaptive.yaml`
   - Increase `entropy_coef` to 0.1
   - Increase `inaction_penalty` in reward config

### Performance Not Improving?

1. **Check if learning rate is too high**:
   - System will auto-reduce if performance plateaus
   - Check `config_adjustments.jsonl` for LR changes

2. **Review reward function**:
   - May need to adjust `pnl_weight` vs penalties
   - Check `logs/adaptive_training/current_reward_config.json`

## Benefits

✅ **No Manual Intervention**: System adjusts automatically  
✅ **Continuous Training**: Never need to stop and restart  
✅ **Intelligent Adaptation**: Adjusts based on actual performance  
✅ **Performance Tracking**: Full history of all evaluations  
✅ **Auto-Save**: Best models saved automatically  

## Next Steps

1. Start training with adaptive config:
   ```bash
   uv run python src/train.py --config configs/train_config_adaptive.yaml --device cuda
   ```

2. Monitor progress:
   - Watch console for adaptive evaluation messages
   - Check `logs/adaptive_training/` for detailed logs

3. Let it run:
   - System will automatically adjust parameters
   - Best models will be saved automatically
   - Training continues until completion

The system is now **fully adaptive** and will intelligently optimize itself during training!

