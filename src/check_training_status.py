"""
Check Training Status and Performance

Analyzes current training progress and model performance.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
import os

def check_model_info(model_path: str):
    """Check basic info about a model file"""
    try:
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"\n📦 Model: {Path(model_path).name}")
        
        if 'actor_state_dict' in checkpoint:
            print("   ✅ Valid PPO checkpoint")
            # Try to get some info about the model
            actor_keys = list(checkpoint['actor_state_dict'].keys())
            if actor_keys:
                print(f"   📊 Actor network layers: {len([k for k in actor_keys if 'weight' in k])}")
        
        if 'actor_optimizer_state_dict' in checkpoint:
            print("   ✅ Optimizer state included")
            
        file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        print(f"   💾 File size: {file_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def get_latest_checkpoint(models_dir: str = "models"):
    """Get the latest checkpoint number"""
    checkpoints = []
    for f in Path(models_dir).glob("checkpoint_*.pt"):
        try:
            step = int(f.stem.split('_')[1])
            checkpoints.append((step, f))
        except:
            continue
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0]
    return None

def analyze_training_progress(config_path: str = "configs/train_config.yaml"):
    """Analyze training progress"""
    import yaml
    
    print("="*70)
    print("TRAINING PROGRESS ANALYSIS")
    print("="*70)
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        target_steps = config['training']['total_timesteps']
        save_freq = config['training']['save_freq']
    except:
        target_steps = 1000000
        save_freq = 10000
        print("⚠️  Could not load config, using defaults")
    
    # Find checkpoints
    models_dir = Path("models")
    all_checkpoints = sorted([f for f in models_dir.glob("checkpoint_*.pt")], 
                            key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0)
    
    if not all_checkpoints:
        print("\n❌ No training checkpoints found!")
        print("   Start training with: python src/train.py")
        return
    
    # Get latest
    latest = get_latest_checkpoint()
    if latest:
        latest_step, latest_path = latest
        progress_pct = (latest_step / target_steps) * 100
        
        print(f"\n📊 Training Progress:")
        print(f"   Latest checkpoint: {latest_step:,} / {target_steps:,} steps")
        print(f"   Progress: {progress_pct:.1f}%")
        print(f"   Remaining: {target_steps - latest_step:,} steps")
        
        # Estimate time remaining (if we have training history)
        num_checkpoints = len(all_checkpoints)
        if num_checkpoints >= 2:
            first_checkpoint = min([int(f.stem.split('_')[1]) for f in all_checkpoints if f.stem.split('_')[1].isdigit()])
            steps_per_checkpoint = (latest_step - first_checkpoint) / max(1, num_checkpoints - 1)
            if steps_per_checkpoint > 0:
                remaining_checkpoints = (target_steps - latest_step) / save_freq
                print(f"   Estimated remaining checkpoints: ~{remaining_checkpoints:.0f}")
    
    # Check best model
    best_model_path = models_dir / "best_model.pt"
    if best_model_path.exists():
        print(f"\n🏆 Best Model:")
        print(f"   ✅ best_model.pt exists")
        check_model_info(str(best_model_path))
        
        # Check if best_model matches latest
        if latest and latest_path.name != "best_model.pt":
            print(f"   📝 Note: Latest checkpoint ({latest_step}) may be newer than best_model")
    
    # List all checkpoints
    print(f"\n📦 Available Checkpoints: {len(all_checkpoints)}")
    if len(all_checkpoints) <= 10:
        for cp in all_checkpoints:
            step = int(cp.stem.split('_')[1]) if cp.stem.split('_')[1].isdigit() else 0
            print(f"   - checkpoint_{step}.pt")
    else:
        # Show first 3, last 3, and latest
        print(f"   First 3: {', '.join([f.stem for f in all_checkpoints[:3]])}")
        print(f"   Last 3: {', '.join([f.stem for f in all_checkpoints[-3:]])}")
        print(f"   Latest: {latest_path.name if latest else 'N/A'}")
    
    # Check TensorBoard logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if log_dirs:
            print(f"\n📈 TensorBoard Logs: {len(log_dirs)} training runs found")
            latest_log = log_dirs[0]
            print(f"   Latest: {latest_log.name}")
            print(f"   View with: tensorboard --logdir {latest_log}")
            
            # Check log file size
            events_files = list(latest_log.glob("events.out.tfevents.*"))
            if events_files:
                log_size = sum(f.stat().st_size for f in events_files) / 1024  # KB
                print(f"   Log size: {log_size:.1f} KB")
    
    # Check for evaluation results
    eval_files = list(Path("logs").glob("*evaluation*.json")) if Path("logs").exists() else []
    if eval_files:
        print(f"\n📊 Evaluation Results: {len(eval_files)} found")
        for ef in eval_files[-3:]:  # Show last 3
            print(f"   - {ef.name}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if latest and latest_step < target_steps * 0.5:
        print(f"   ⚠️  Training is less than 50% complete")
        print(f"      Consider continuing training or reviewing early results")
    
    if latest and latest_step >= target_steps * 0.9:
        print(f"   ✅ Training is nearly complete!")
        print(f"      Run evaluation: python src/backtest.py --model models/best_model.pt")
    
    if not best_model_path.exists() and latest:
        print(f"   ⚠️  No best_model.pt found")
        print(f"      Latest checkpoint may need to be copied or training may not have completed an episode")
    
    print("\n" + "="*70)

def check_performance_metrics():
    """Check if we can extract performance metrics"""
    print("\n📊 Performance Metrics:")
    
    # Try to load best model and get info
    best_model = Path("models/best_model.pt")
    if best_model.exists():
        try:
            # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
            checkpoint = torch.load(str(best_model), map_location='cpu', weights_only=False)
            print("   ✅ Best model loaded successfully")
            # Model structure is there, but we'd need to evaluate to get performance
            print("   💡 Run backtest to get performance metrics:")
            print("      python src/backtest.py --model models/best_model.pt")
        except Exception as e:
            print(f"   ⚠️  Could not load model: {e}")
    else:
        latest = get_latest_checkpoint()
        if latest:
            _, latest_path = latest
            print(f"   💡 Use latest checkpoint for evaluation:")
            print(f"      python src/backtest.py --model {latest_path}")

if __name__ == "__main__":
    analyze_training_progress()
    check_performance_metrics()
    
    print("\n🔍 For detailed performance analysis:")
    print("   1. View TensorBoard: tensorboard --logdir logs")
    print("   2. Run backtest: python src/backtest.py --model models/best_model.pt")
    print("   3. Compare models: Use src/model_evaluation.py")

