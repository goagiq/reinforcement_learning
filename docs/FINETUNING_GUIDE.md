# DeepSeek-R1:8b Fine-Tuning Guide

## Overview

This guide explains how to fine-tune DeepSeek-R1:8b for continuous learning in the trading strategy context.

## Why Fine-Tune DeepSeek-R1?

- **Domain Adaptation**: Better understanding of trading terminology and concepts
- **Task-Specific Learning**: Learn from your actual trading outcomes
- **Improved Reasoning**: Better reasoning chains for trading decisions
- **Faster Responses**: More focused responses with less irrelevant reasoning

## Fine-Tuning Approaches

### Option 1: LoRA (Low-Rank Adaptation) ⭐ Recommended

**Pros:**
- Efficient: Only trains 1-5% of parameters
- Fast: Quick training cycles
- Low memory: Works with consumer GPUs
- Preserves base model knowledge

**Best for:** Regular fine-tuning with new trading data

### Option 2: QLoRA (Quantized LoRA)

**Pros:**
- Even more efficient than LoRA
- Works with limited GPU memory
- Faster training

**Best for:** Systems with limited GPU memory

### Option 3: Full Fine-Tuning

**Pros:**
- Maximum adaptation
- Best performance potential

**Cons:**
- Very resource intensive
- Risk of catastrophic forgetting
- Requires significant GPU memory

**Best for:** Major model updates (not recommended for continuous learning)

## Data Format

Training data should be in JSON format:

```json
[
  {
    "instruction": "Analyze this trading scenario and provide a recommendation.",
    "input": "Market: trending up, Volume: high, RSI: 65, Action: Buy @ 0.8 position size",
    "output": "This appears to be a good trading opportunity. The market is in an uptrend with high volume supporting the move. The RSI at 65 indicates strength without being overbought. The position size of 0.8 is appropriate for this high-confidence setup. Risk factors to monitor: watch for volume divergence and consider taking profits if RSI exceeds 75."
  },
  {
    "instruction": "Analyze why this trade failed.",
    "input": "Market: ranging, Volume: declining, Action: Buy @ 0.6, Outcome: -2% loss",
    "output": "This trade failed because it was taken in a ranging market without clear direction. The declining volume indicated lack of conviction. Entry was premature - should have waited for a clear breakout or breakdown. Lesson: Avoid trading in ranging markets without clear volume confirmation."
  }
]
```

## Generating Training Data

Use the continuous learning pipeline to generate training data:

```python
from src.continuous_learning import ContinuousLearningPipeline, ExperienceBuffer

# Generate training data from experiences
pipeline = ContinuousLearningPipeline(config, buffer, agent, reasoning_engine)
training_data = pipeline.generate_training_data_for_deepseek()

# Save
import json
with open("data/finetuning/deepseek_training.json", "w") as f:
    json.dump(training_data, f, indent=2)
```

## Fine-Tuning with Unsloth (Recommended)

### 1. Install Dependencies

```bash
pip install unsloth transformers datasets accelerate bitsandbytes
```

### 2. Fine-Tuning Script

Create `scripts/finetune_deepseek.py`:

```python
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Prepare for LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# Load your training data
dataset = load_dataset("json", data_files="data/finetuning/deepseek_training.json")

# Fine-tune
model = FastLanguageModel.get_peft_model(model, r = 16, ...)
model.train()

# Save
model.save_pretrained("models/deepseek-r1-finetuned")
```

## Fine-Tuning with Ollama

### Option 1: Create Custom Modelfile

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM deepseek-r1:8b

PARAMETER temperature 0.3
PARAMETER top_p 0.9
EOF

# Create custom model
ollama create deepseek-r1-trading -f Modelfile
```

### Option 2: Use Ollama's Fine-Tuning API

Ollama doesn't directly support fine-tuning yet, but you can:
1. Fine-tune using Unsloth/PEFT
2. Export model to GGUF format
3. Import into Ollama

## Integration

After fine-tuning, update the reasoning engine:

```python
# In src/reasoning_engine.py or config
self.model = "deepseek-r1-trading"  # Use fine-tuned model
```

## Training Schedule

**Recommended Schedule:**
- **Weekly**: Fine-tune with new experiences from the past week
- **Monthly**: Full evaluation and comparison
- **Quarterly**: Major model update if performance improves significantly

**Trigger Conditions:**
- After collecting 100+ new annotated experiences
- When win rate drops below threshold
- After significant market regime change

## Evaluation

Compare fine-tuned vs base model:

```python
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator(config)

# Test base model
base_metrics = evaluator.evaluate_model("models/base_reasoning_model")

# Test fine-tuned model  
finetuned_metrics = evaluator.evaluate_model("models/finetuned_reasoning_model")

# Compare
if finetuned_metrics.sharpe_ratio > base_metrics.sharpe_ratio * 1.05:
    print("✅ Fine-tuned model is better - deploy it")
else:
    print("⚠️  Fine-tuned model not better - keep base model")
```

## Best Practices

1. **Start Small**: Begin with small datasets (50-100 examples)
2. **Validate**: Always evaluate before deploying
3. **Version Control**: Use model versioning system
4. **Rollback Ready**: Keep base model available
5. **Monitor**: Track fine-tuned model performance vs base
6. **Iterate**: Regular small updates better than infrequent large updates

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Ollama Model Creation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

