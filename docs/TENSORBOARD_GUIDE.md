# TensorBoard Guide - Viewing Training Metrics

## Quick Start

### Method 1: Command Line (Recommended)

```bash
# Activate your virtual environment first
.venv\Scripts\activate  # Windows

# Run TensorBoard
tensorboard --logdir logs

# Or with specific port
tensorboard --logdir logs --port 6006
```

### Method 2: With Full Path

If you're not in the project directory:

```bash
cd D:\NT8-RL
tensorboard --logdir logs
```

### Method 3: View Specific Training Run

To view a specific training session:

```bash
tensorboard --logdir logs\ppo_training_20251031_183437
```

## Accessing TensorBoard

Once you run the command, you'll see:

```
TensorBoard 2.x.x at http://localhost:6006/
```

**Then:**
1. Open your web browser
2. Go to: `http://localhost:6006`
3. You'll see all your training metrics!

## What You'll See in TensorBoard

### Main Tabs

1. **SCALARS** - Training metrics over time
   - `train/loss` - Overall training loss
   - `train/policy_loss` - Policy network loss
   - `train/value_loss` - Value network loss
   - `train/entropy` - Exploration entropy
   - `episode/reward` - Episode rewards
   - `episode/pnl` - Profit and loss
   - `episode/trades` - Number of trades
   - `episode/equity` - Account equity
   - `eval/mean_reward` - Evaluation rewards

2. **GRAPHS** - Neural network architecture visualization

3. **IMAGES** - If you log any images

4. **HISTOGRAMS** - Parameter distributions

## Key Metrics to Monitor

### For Your Current Stage (21% = 210k steps)

**Look for these trends:**

1. **train/loss** 📉
   - Should be **decreasing** over time
   - Current value: Check latest point
   - Good sign: Steady downward trend

2. **train/policy_loss** 📉
   - Should be **very low** (< 0.001)
   - Current: Likely around 0.0002-0.001
   - Good sign: Low and stable

3. **train/value_loss** 📉
   - Should be **decreasing**
   - Current: May be around 0.1-0.2
   - Good sign: Downward trend

4. **episode/reward** 📈
   - May still be **negative or mixed** at 21%
   - Should start improving around 300k steps
   - Expected: Gradually climbing toward positive

5. **episode/pnl** 💰
   - May be negative at this stage
   - Should improve as training progresses
   - Target: Positive by 300k+ steps

## Troubleshooting

### TensorBoard Not Starting

```bash
# Check if TensorBoard is installed
pip install tensorboard

# Check if logs directory exists
dir logs

# Try with explicit Python path
python -m tensorboard --logdir logs
```

### Port Already in Use

```bash
# Use different port
tensorboard --logdir logs --port 6007

# Or find and kill process using port 6006
netstat -ano | findstr :6006
taskkill /PID <process_id> /F
```

### No Data Showing

**Possible reasons:**
1. Logs directory empty or wrong path
2. Training hasn't generated logs yet
3. Using wrong log directory

**Solutions:**
```bash
# Check what's in logs
dir logs

# View specific training run
tensorboard --logdir logs\ppo_training_20251031_183437

# Check latest log directory
tensorboard --logdir logs --reload_interval 5
```

### Windows PowerShell Issues

If you have issues in PowerShell:

```powershell
# Use cmd.exe instead
cmd
cd D:\NT8-RL
.venv\Scripts\activate
tensorboard --logdir logs
```

Or use Git Bash (which you have):

```bash
# In Git Bash
cd /d/NT8-RL
source .venv/Scripts/activate
tensorboard --logdir logs
```

## Advanced Usage

### Auto-Reload

TensorBoard auto-reloads, but you can set interval:

```bash
tensorboard --logdir logs --reload_interval 5  # Reload every 5 seconds
```

### Compare Multiple Runs

View all training runs at once:

```bash
tensorboard --logdir logs
```

Then use the checkboxes in TensorBoard UI to compare different runs.

### Export Data

You can also extract metrics programmatically:

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/ppo_training_20251031_183437')
ea.Reload()

# Get scalar data
scalars = ea.Scalars('train/loss')
for scalar in scalars:
    print(f"Step {scalar.step}: {scalar.value}")
```

## Quick Reference Commands

```bash
# Basic
tensorboard --logdir logs

# Specific port
tensorboard --logdir logs --port 6007

# Specific run
tensorboard --logdir logs\ppo_training_20251031_183437

# Auto-reload
tensorboard --logdir logs --reload_interval 2

# Host from other machines
tensorboard --logdir logs --host 0.0.0.0
```

## What to Look For - Training Health

### ✅ Good Signs (Your Training Should Show)
- Loss decreasing over time
- Policy loss very low
- Value loss decreasing
- Episodes completing (even if rewards are mixed)
- No sudden spikes or crashes

### ⚠️ Warning Signs
- Loss increasing
- NaN values (model crashed)
- Loss not changing (learning stalled)
- Very high variance in rewards (might need hyperparameter tuning)

### 📊 Expected at Your Stage (210k steps)
- Loss: Decreasing from initial values
- Policy Loss: < 0.001
- Rewards: Mixed (negative to small positive)
- Episodes: Long (this is normal for trading)

## Next Steps

1. **Start TensorBoard**: `tensorboard --logdir logs`
2. **Open Browser**: Go to http://localhost:6006
3. **Check SCALARS tab**: Review training metrics
4. **Monitor Trends**: Watch for improvements
5. **Take Screenshots**: Document progress at milestones

---

**Tip**: Keep TensorBoard running while training to see real-time updates!

