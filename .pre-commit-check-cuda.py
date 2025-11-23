#!/usr/bin/env python3
"""
Pre-commit hook to ensure CUDA PyTorch is installed
Run this before any commit to prevent CPU-only PyTorch
"""

import sys
import subprocess

def check_cuda():
    try:
        import torch
        version = torch.__version__
        has_cuda_build = '+cu' in version
        cuda_available = torch.cuda.is_available()
        
        if not has_cuda_build:
            print("[ERROR] CPU-only PyTorch detected!")
            print(f"  Version: {version}")
            print("  Run: python ensure_cuda_pytorch.py")
            return False
        
        if not cuda_available:
            print("[WARN] CUDA PyTorch installed but CUDA runtime not available")
            print("  Check NVIDIA drivers")
            return True  # Don't fail commit, just warn
        
        print(f"[OK] CUDA PyTorch: {version}, GPU available")
        return True
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return False

if __name__ == "__main__":
    if not check_cuda():
        sys.exit(1)

