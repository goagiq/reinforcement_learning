# Model Architecture Mismatch & Fallback Mechanism - Detailed Explanation

## What Happened?

When you tried to run RL agent backtesting, you saw this warning:

```
RL agent backtest failed: Error(s) in loading state_dict for ActorNetwork:
    size mismatch for feature_layers.0.weight: copying a param with shape 
    torch.Size([128, 900]) from checkpoint, the shape in current model is 
    torch.Size([256, 900]).
```

This means **the saved model was trained with a different neural network architecture** than what the current code expects.

## Understanding Neural Network Architecture

### What is "Architecture"?

The architecture defines the structure of the neural network:
- **Number of layers**
- **Number of neurons in each layer** (hidden dimensions)
- **Connection patterns**

Think of it like a building blueprint - you can't load furniture from a 2-bedroom house into a 3-bedroom house.

### Current Code Architecture

**Default Architecture (from `src/rl_agent.py`):**
```python
hidden_dims = [256, 256, 128]  # Default if not specified
```

This creates:
- **Layer 1**: 900 inputs → **256 neurons**
- **Layer 2**: 256 inputs → **256 neurons**
- **Layer 3**: 256 inputs → **128 neurons**
- **Output**: 128 inputs → 1 neuron (action)

**Total Parameters:** ~350,000 weights

### Saved Model Architecture

**From the error message, the saved model has:**
```python
hidden_dims = [128, 128, 64]  # Different architecture
```

This creates:
- **Layer 1**: 900 inputs → **128 neurons** ❌ (current code expects 256)
- **Layer 2**: 128 inputs → **128 neurons** ❌ (current code expects 256)
- **Layer 3**: 128 inputs → **64 neurons** ❌ (current code expects 128)
- **Output**: 64 inputs → 1 neuron ❌ (current code expects 128)

**Total Parameters:** ~140,000 weights (smaller model)

## Why Does This Happen?

### Common Scenarios:

1. **Code Evolution**: Architecture was changed to improve performance
   - Smaller model → Larger model (more capacity)
   - Larger model → Smaller model (faster inference)

2. **Different Training Runs**: Model was trained with different config
   - Old config: `hidden_dims: [128, 128, 64]`
   - New config: `hidden_dims: [256, 256, 128]` (default)

3. **Hyperparameter Tuning**: Architecture was adjusted during development
   - Experimented with different sizes
   - Forgot to update saved model

4. **Different Versions**: Code was updated but model wasn't retrained
   - Model trained with v1.0
   - Code updated to v2.0 with new architecture

## The Fallback Mechanism

### How It Works

**Location:** `src/scenario_simulator.py` lines 556-580

```python
try:
    # Try to load and use RL agent
    agent.load(actual_model_path)
    # ... run backtest with RL agent ...
except Exception as e:
    # If RL agent fails, fall back to simple backtest
    warnings.warn(f"RL agent backtest failed: {e}. Falling back to simple backtest.")
    
    # Use simple backtest instead
    returns = price_data['close'].pct_change().fillna(0)
    equity_curve = initial_capital * (1 + returns).cumprod()
    # ... calculate metrics ...
```

### What the Fallback Does

**Simple Backtest (fallback):**
1. Calculates price returns from data
2. Simulates trades by buying every 20 bars and selling 10 bars later
3. Calculates equity curve and basic metrics
4. **Does NOT use RL agent** - just simulates simple buy/hold strategy

**Limitations:**
- ❌ No intelligent decision making
- ❌ No short positions
- ❌ No risk management
- ❌ No trend following
- ✅ Still provides scenario testing (but less realistic)

### Why Fallback Exists

**Graceful Degradation:**
- System continues to work even if model unavailable
- Provides basic scenario testing without RL agent
- Allows testing while model is being retrained

**Error Prevention:**
- Prevents crashes when architecture mismatches
- Allows development/testing without trained model
- Makes system more robust

## Solutions

### Option 1: Retrain Model with Current Architecture ⭐ RECOMMENDED

**Best long-term solution:**

1. **Check current architecture:**
   ```python
   # In src/rl_agent.py, default is:
   hidden_dims = [256, 256, 128]
   ```

2. **Start training:**
   ```bash
   python src/train.py --config configs/train_config.yaml
   ```

3. **Model will be saved with correct architecture**

4. **Benefits:**
   - ✅ Uses latest architecture (potentially better)
   - ✅ Fully compatible with current code
   - ✅ Can use RL agent backtesting

**Time Required:** Depends on training steps (hours to days)

### Option 2: Update Code to Match Saved Model

**If you want to use existing model without retraining:**

1. **Find saved model architecture:**
   ```python
   # Check checkpoint metadata
   import torch
   checkpoint = torch.load("models/best_model.pt", map_location='cpu')
   print(checkpoint.get('hidden_dims', 'Not found'))
   ```

2. **Update code to match:**
   ```python
   # In src/rl_agent.py or config file
   hidden_dims = [128, 128, 64]  # Match saved model
   ```

3. **Update scenario simulator:**
   ```python
   # In src/scenario_simulator.py
   agent = PPOAgent(
       state_dim=env.state_dim,
       action_range=tuple(action_range),
       device="cpu",
       hidden_dims=[128, 128, 64]  # Match saved model
   )
   ```

**Benefits:**
   - ✅ Can use existing trained model immediately
   - ✅ No retraining needed

**Drawbacks:**
   - ❌ May miss improvements from newer architecture
   - ❌ Need to maintain old architecture in code

### Option 3: Use Fallback (Current Behavior)

**If you don't need RL agent backtesting right now:**

- System already works with fallback
- Provides basic scenario testing
- Can retrain model later when ready

**Benefits:**
   - ✅ No immediate action needed
   - ✅ System continues to work

**Drawbacks:**
   - ❌ Less realistic scenario testing
   - ❌ No RL agent intelligence

## How to Check Your Model Architecture

### Method 1: Check Checkpoint Metadata

```python
import torch
from pathlib import Path

model_path = Path("models/best_model.pt")
if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("Checkpoint keys:", checkpoint.keys())
    print("Hidden dims:", checkpoint.get('hidden_dims', 'Not saved'))
    print("State dim:", checkpoint.get('state_dim', 'Not saved'))
    
    # Check actual layer sizes
    if 'actor_state_dict' in checkpoint:
        actor = checkpoint['actor_state_dict']
        for key, value in actor.items():
            if 'weight' in key and 'feature_layers' in key:
                print(f"{key}: {value.shape}")
```

### Method 2: Check Training Config

```bash
# Check config file used for training
cat configs/train_config.yaml | grep -i hidden
```

### Method 3: Inspect Model File

```python
# Load and inspect model architecture
from src.rl_agent import PPOAgent

# Current code default
agent = PPOAgent(state_dim=900, device='cpu')
print("Current default architecture:")
for i, layer in enumerate(agent.actor.feature_layers):
    if isinstance(layer, torch.nn.Linear):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")
```

## Recommendations

### For Production Use:

1. **Retrain model** with current architecture
2. **Save architecture** in checkpoint metadata (already done)
3. **Load architecture** from checkpoint when loading model
4. **Version control** architecture changes

### For Development/Testing:

1. **Use fallback** for now (current behavior)
2. **Retrain when ready** to use RL agent
3. **Document architecture changes** in code comments

## Code Improvement Suggestions

### Better Error Handling

```python
# In src/scenario_simulator.py
try:
    # Try to load with saved architecture
    checkpoint = torch.load(actual_model_path, map_location='cpu')
    saved_hidden_dims = checkpoint.get('hidden_dims', [256, 256, 128])
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_range=tuple(action_range),
        device="cpu",
        hidden_dims=saved_hidden_dims  # Use saved architecture
    )
    agent.load(actual_model_path)
except Exception as e:
    # Fallback to simple backtest
    warnings.warn(f"RL agent backtest failed: {e}. Falling back to simple backtest.")
    # ... fallback code ...
```

This would automatically detect and use the saved model's architecture!

## Summary

| Aspect | Current | Saved Model |
|--------|---------|-------------|
| Layer 1 | 256 neurons | 128 neurons |
| Layer 2 | 256 neurons | 128 neurons |
| Layer 3 | 128 neurons | 64 neurons |
| Output | 128 → 1 | 64 → 1 |
| Parameters | ~350K | ~140K |

**Result:** Architecture mismatch → Fallback to simple backtest

**Solution:** Retrain model with current architecture OR update code to match saved model

**Status:** System works with fallback, but RL agent backtesting is unavailable until architecture matches

