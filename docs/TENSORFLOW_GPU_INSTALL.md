# TensorFlow GPU Installation Guide

## Your System Configuration

- **GPU**: NVIDIA GeForce RTX 2070
- **CUDA Version**: 12.2
- **Python**: 3.13.5
- **Platform**: Windows

## Important Note

⚠️ **This project uses PyTorch, not TensorFlow!**

- Your RL training uses **PyTorch** (already installed)
- TensorBoard (for visualization) is separate from TensorFlow
- TensorFlow GPU is **only needed if you plan to use TensorFlow models**

If you just need TensorBoard visualization (which you do), you can use:
```bash
pip install tensorboard  # Already in requirements.txt
```

This doesn't require TensorFlow at all.

---

## If You Still Want TensorFlow GPU

TensorFlow may have compatibility issues with Python 3.13 (very new). Here are options:

### Option 1: TensorFlow 2.15+ (Recommended if compatible)

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Or specific version
pip install tensorflow==2.15.0
```

### Option 2: Nightly Build (May support Python 3.13)

```bash
pip install tf-nightly
```

### Option 3: Check Compatibility First

```bash
# Check if TensorFlow supports Python 3.13
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')"
# If TensorFlow doesn't support 3.13, you may need Python 3.11 or 3.12
```

---

## Verification Steps

After installation, verify GPU support:

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
```

---

## Alternative: Use PyTorch (What You're Already Using)

Since this project uses PyTorch, you already have GPU support:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

Your training should already be using GPU if configured with `--device cuda`.

---

## Recommendation

**For this project**: You don't need TensorFlow! Just use:
- **PyTorch** (already installed) for training
- **TensorBoard** (already in requirements) for visualization

TensorBoard works independently and doesn't need TensorFlow.

