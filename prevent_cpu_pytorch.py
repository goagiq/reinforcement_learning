#!/usr/bin/env python3
"""
Prevent CPU-only PyTorch installation
This script should be run before uv sync or pip install to ensure CUDA PyTorch is used
"""

import sys
import subprocess
import os
from pathlib import Path

def check_pyproject_toml():
    """Verify pyproject.toml has CUDA configuration"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("[ERROR] pyproject.toml not found")
        return False
    
    content = pyproject_path.read_text()
    
    # Check for CUDA index
    if "download.pytorch.org/whl/cu121" not in content:
        print("[ERROR] pyproject.toml missing CUDA index configuration")
        print("  Add: extra-index-url = [\"https://download.pytorch.org/whl/cu121\"]")
        return False
    
    # Check for index-strategy
    if "index-strategy" not in content or "unsafe-best-match" not in content:
        print("[WARN] pyproject.toml may not prefer CUDA builds")
        print("  Add: index-strategy = \"unsafe-best-match\"")
        return False
    
    print("[OK] pyproject.toml has CUDA configuration")
    return True

def check_current_pytorch():
    """Check if current PyTorch is CUDA-enabled"""
    try:
        import torch
        version = torch.__version__
        has_cuda_build = '+cu' in version
        cuda_available = torch.cuda.is_available()
        
        if not has_cuda_build:
            print(f"[ERROR] CPU-only PyTorch detected: {version}")
            print("  Run: python ensure_cuda_pytorch.py")
            return False
        
        if cuda_available:
            print(f"[OK] CUDA PyTorch installed: {version}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"[WARN] CUDA PyTorch installed but CUDA runtime not available: {version}")
            return True  # Still OK, just runtime issue
    except ImportError:
        print("[INFO] PyTorch not yet installed")
        return True  # OK, will be installed

def main():
    """Main check function"""
    print("="*70)
    print("CUDA PyTorch Protection Check")
    print("="*70)
    print()
    
    # Check pyproject.toml configuration
    if not check_pyproject_toml():
        print()
        print("[ERROR] Configuration issue - fix before running uv sync")
        sys.exit(1)
    
    print()
    
    # Check current installation
    if not check_current_pytorch():
        print()
        print("[ERROR] CPU-only PyTorch detected - fix before continuing")
        print("  Run: python ensure_cuda_pytorch.py")
        sys.exit(1)
    
    print()
    print("[OK] All checks passed - safe to run uv sync")
    sys.exit(0)

if __name__ == "__main__":
    main()

