"""
Automated Training Resume Script

Finds the latest checkpoint automatically and resumes training from it.
This makes resuming training as easy as one command.

Usage:
    python resume_training.py
    python resume_training.py --device cuda
    python resume_training.py --device cuda --config configs/train_config_gpu_optimized.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml


def find_latest_checkpoint(models_dir: Path = Path("models")) -> Path:
    """
    Find the latest checkpoint file.
    
    Args:
        models_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not models_dir.exists():
        return None
    
    checkpoints = list(models_dir.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(
        description="Automatically resume training from latest checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default config and auto-detect device
    python resume_training.py
    
    # Specify GPU training
    python resume_training.py --device cuda
    
    # Use specific config
    python resume_training.py --config configs/train_config_gpu_optimized.yaml --device cuda
    
    # Check for checkpoints without resuming
    python resume_training.py --check-only
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config_gpu_optimized.yaml",
        help="Path to training config file (default: configs/train_config_gpu_optimized.yaml)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda). If not specified, will use config default"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for checkpoints, don't resume training"
    )
    
    args = parser.parse_args()
    
    # Find latest checkpoint
    print("="*80)
    print("NT8 RL Training - Automatic Resume")
    print("="*80)
    print()
    print("🔍 Searching for checkpoints...")
    
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"✅ Found latest checkpoint: {latest_checkpoint.name}")
        print(f"   Path: {latest_checkpoint}")
        print(f"   Modified: {latest_checkpoint.stat().st_mtime}")
        print()
    else:
        print("❌ No checkpoints found in models/ directory")
        print()
        print("This means:")
        print("  - Either training has never been run")
        print("  - Or training hasn't reached 10,000 timesteps yet")
        print()
        print("Starting fresh training instead...")
        print()
        
        # Start fresh training
        import subprocess
        cmd = ["python", "src/train.py", "--config", args.config]
        if args.device:
            cmd.extend(["--device", args.device])
        
        print(f"Running: {' '.join(cmd)}")
        print()
        
        if args.check_only:
            print("(--check-only specified, not actually running)")
            sys.exit(0)
        
        sys.exit(subprocess.call(cmd))
    
    if args.check_only:
        print("Checkpoint found successfully.")
        print("To resume, run without --check-only flag.")
        sys.exit(0)
    
    # Load config to get default device if not specified
    device = args.device
    if not device:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                device = config.get("training", {}).get("device", "cpu")
        except Exception as e:
            print(f"⚠️  Could not load config, defaulting to CPU: {e}")
            device = "cpu"
    
    # Display resume information
    print("📋 Resume Information:")
    print(f"   Config: {args.config}")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {latest_checkpoint}")
    print()
    
    # Check CUDA availability if requesting GPU
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU available: {gpu_name}")
            else:
                print("⚠️  CUDA requested but not available. Will use CPU instead.")
                device = "cpu"
        except ImportError:
            print("⚠️  PyTorch not found. Will attempt anyway...")
    
    print()
    print("="*80)
    print("🚀 Resuming Training...")
    print("="*80)
    print()
    
    # Import and call train script with checkpoint
    try:
        from src.train import Trainer, load_config
        
        # Load config
        config = load_config(args.config)
        
        # Override device if specified
        if args.device:
            config["training"]["device"] = args.device
        else:
            config["training"]["device"] = device
        
        # Create trainer with checkpoint
        trainer = Trainer(config, checkpoint_path=str(latest_checkpoint))
        
        # Resume training
        trainer.train()
        
    except KeyboardInterrupt:
        print()
        print()
        print("Training stopped by user.")
        print("You can resume again with: python resume_training.py")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"❌ Error resuming training: {e}")
        print()
        print("Trying to run via CLI instead...")
        print()
        
        # Fall back to CLI
        import subprocess
        cmd = [
            "python", "src/train.py",
            "--config", args.config,
            "--checkpoint", str(latest_checkpoint)
        ]
        if device:
            cmd.extend(["--device", device])
        
        print(f"Running: {' '.join(cmd)}")
        print()
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()








