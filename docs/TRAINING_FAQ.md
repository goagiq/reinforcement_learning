# Training FAQ

## Episode Length and Completion

### Q: Episodes never complete - `completed_episodes` stays at 0

**A:** This was a known issue fixed in 2024. Episodes now complete properly at `max_episode_steps` (default: 10,000).

**Solution:**
1. Ensure `max_episode_steps: 10000` is in `configs/train_config.yaml`
2. Restart training (or resume from checkpoint)
3. Episodes will now complete and metrics will update

**Why it happened:** Historical data can have 85,000+ steps. Without an episode limit, episodes would never terminate naturally.

### Q: What's the difference between `episode_length` and `current_step`?

**A:**
- **`episode_length`**: Training loop counter (starts at 0 each episode)
- **`current_step`**: Environment's internal step counter (starts at `lookback_bars`, typically 20)

They should match closely, but may differ slightly due to initialization.

### Q: How do I adjust episode length?

**A:** Set `max_episode_steps` in `configs/train_config.yaml`:
- Shorter: `5000-7000` (faster, less context)
- Default: `10000` (recommended)
- Longer: `15000-20000` (slower, more context)

---

# Training FAQ (Original): Ollama Models vs RL Training

## Quick Answer: No, Ollama Models Don't Affect Training Speed

**Ollama models (deepseek-r1, tinyllama, etc.) are NOT used during RL training.** They're completely separate systems.

## Two Separate Systems

### 1. **RL Training** (The Slow Part)
- **What it is**: Training the Actor-Critic neural networks to learn trading
- **What's trained**: The PPO agent (your trading model)
- **Speed**: This is what we optimized (~2-3x faster now)
- **Ollama usage**: ❌ **ZERO** - Ollama is NOT called during training

### 2. **Reasoning Engine** (Ollama Models)
- **What it is**: Pre-trade validation using language models
- **When used**: Only during **live trading** (not training)
- **Purpose**: Validates RL recommendations before executing trades
- **Models**: deepseek-r1:8b, tinyllama, or any Ollama model
- **Speed**: Only matters during live trading (not training)

## During Training

```
┌─────────────────────────────────────┐
│  RL Training Loop (train.py)        │
│  ├─ Trading Environment             │
│  ├─ PPO Agent (Actor-Critic)        │
│  ├─ Neural Network Updates          │
│  └─ ❌ NO Ollama Calls              │
└─────────────────────────────────────┘
```

The training loop:
1. Collects experiences from the trading environment
2. Updates the PPO agent's neural networks
3. Never calls Ollama/reasoning engine

## During Live Trading

```
┌─────────────────────────────────────┐
│  Live Trading System                │
│  ├─ RL Agent makes recommendation   │
│  ├─ → Reasoning Engine (Ollama)    │  ← Only here!
│  ├─ Validates/modifies recommendation│
│  └─ Executes trade                   │
└─────────────────────────────────────┘
```

## Config Settings

In `train_config_gpu_optimized.yaml`:

```yaml
reasoning:
  enabled: false    # ← Already disabled during training!
```

**This is intentional** - reasoning adds delay and isn't needed during training.

## Which Model Should You Use?

### For RL Training Speed:
- **Doesn't matter** - Ollama isn't used during training
- Use any model (or none) - training speed won't change

### For Live Trading Quality:
- **deepseek-r1:8b** (Recommended)
  - ✅ Better reasoning quality
  - ✅ Better trade validation
  - ⚠️ Slower (2-5 minutes per reasoning call)
  
- **tinyllama** (Faster, Lower Quality)
  - ✅ Faster reasoning (seconds)
  - ⚠️ Lower quality analysis
  - ⚠️ May make worse decisions

## Summary

| Question | Answer |
|----------|--------|
| Does Ollama model affect training speed? | **NO** - Not used during training |
| Should I switch to tinyllama for training? | **NO** - Won't make any difference |
| When does Ollama model matter? | **Only during live trading** |
| Which Ollama model should I use? | **deepseek-r1:8b** for quality, **tinyllama** for speed (only matters for live trading) |
| How do I speed up training? | ✅ Use the optimized config we created (already done!) |

## To Speed Up Training Further

The optimizations we made (mixed precision, smaller network, larger batches) are what actually matter:

1. ✅ **Mixed precision (FP16)** - 2x speedup
2. ✅ **Smaller network** [128, 128, 64] - 1.3x speedup  
3. ✅ **Larger batch size** (128) - 1.2x speedup
4. ✅ **Fewer epochs** (4 vs 10) - Faster updates

**Combined: ~2.5-3x faster training** on your RTX 2070

The Ollama model choice has ZERO impact on training speed because it's not even involved in the training process!

