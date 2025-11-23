#!/usr/bin/env python3
"""
Ensure CUDA PyTorch is installed and prevent CPU-only version
This script checks and fixes PyTorch installation to use CUDA builds
"""

import subprocess
import sys
import os
from pathlib import Path

def check_cuda_pytorch():
    """Check if CUDA PyTorch is installed"""
    try:
        import torch
        version = torch.__version__
        has_cuda = torch.cuda.is_available()
        has_cuda_build = '+cu' in version or (hasattr(torch.version, 'cuda') and torch.version.cuda is not None)
        
        return {
            "installed": True,
            "version": version,
            "has_cuda_build": has_cuda_build,
            "cuda_available": has_cuda,
            "gpu_name": torch.cuda.get_device_name(0) if has_cuda else None
        }
    except ImportError:
        return {"installed": False}
    except Exception as e:
        return {"installed": True, "error": str(e)}

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch"""
    print("Installing CUDA-enabled PyTorch...")
    
    # Detect if using uv or pip
    use_uv = False
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, timeout=2)
        if result.returncode == 0:
            use_uv = True
    except:
        pass
    
    # Uninstall existing PyTorch
    print("  Uninstalling existing PyTorch packages...")
    if use_uv:
        subprocess.run(["uv", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      capture_output=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                      capture_output=True)
    
    # Install CUDA PyTorch
    print("  Installing CUDA-enabled PyTorch (cu121)...")
    if use_uv:
        result = subprocess.run([
            "uv", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], capture_output=True, text=True)
    else:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [ERROR] Installation failed: {result.stderr}")
        return False
    
    print("  [OK] CUDA PyTorch installed")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensure CUDA PyTorch is installed")
    parser.add_argument("--auto", action="store_true", help="Auto-install without prompting")
    args = parser.parse_args()
    
    print("="*70)
    print("CUDA PyTorch Verification Script")
    print("="*70)
    print()
    
    status = check_cuda_pytorch()
    
    if not status.get("installed"):
        print("[ERROR] PyTorch is not installed")
        if not args.auto:
            response = input("Install CUDA PyTorch now? [Y/n]: ").strip().lower()
            if response and response != 'y' and response != 'yes':
                sys.exit(1)
        if install_cuda_pytorch():
            status = check_cuda_pytorch()
        else:
            sys.exit(1)
    
    if not status.get("has_cuda_build"):
        print("[WARN] PyTorch is CPU-only build")
        print(f"  Current version: {status.get('version', 'unknown')}")
        if not args.auto:
            response = input("Reinstall with CUDA support? [Y/n]: ").strip().lower()
            if response and response != 'y' and response != 'yes':
                print("[WARN] Continuing with CPU-only PyTorch")
                sys.exit(0)
        
        if install_cuda_pytorch():
            status = check_cuda_pytorch()
        else:
            sys.exit(1)
    
    if status.get("cuda_available"):
        print("[OK] CUDA PyTorch is installed and working!")
        print(f"  Version: {status.get('version')}")
        print(f"  GPU: {status.get('gpu_name')}")
        sys.exit(0)
    else:
        print("[WARN] CUDA PyTorch is installed but CUDA runtime not available")
        print(f"  Version: {status.get('version')}")
        print("  Check NVIDIA drivers and CUDA toolkit installation")
        sys.exit(1)

if __name__ == "__main__":
    main()

