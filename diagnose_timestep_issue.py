"""
Diagnose why timesteps remain at 0 during training.

This script checks:
1. If timestep is being incremented in the training loop
2. If there are exceptions causing resets
3. If the trainer object is being recreated
4. Check backend logs for timestep values
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_backend_logs():
    """Check backend logs for timestep information"""
    print("\n[1] Checking Backend Logs for Timestep Information...")
    print("-" * 70)
    
    log_dir = project_root / "logs"
    if not log_dir.exists():
        print(f"  [WARN] Logs directory not found: {log_dir}")
        return
    
    # Look for recent log files
    log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.txt"))
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_files:
        print(f"  [WARN] No log files found in {log_dir}")
        return
    
    print(f"  [OK] Found {len(log_files)} log file(s)")
    
    # Check most recent log file
    recent_log = log_files[0]
    print(f"  [INFO] Checking most recent log: {recent_log.name}")
    
    try:
        with open(recent_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Look for timestep mentions
        timestep_lines = [line for line in lines if 'timestep' in line.lower() or 'step' in line.lower()]
        
        if timestep_lines:
            print(f"  [OK] Found {len(timestep_lines)} lines mentioning timestep")
            print(f"  [INFO] Last 5 timestep-related lines:")
            for line in timestep_lines[-5:]:
                print(f"    {line.strip()[:100]}")
        else:
            print(f"  [WARN] No timestep mentions found in log")
        
        # Look for errors or exceptions
        error_lines = [line for line in lines if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower()]
        if error_lines:
            print(f"\n  [WARN] Found {len(error_lines)} error/exception lines")
            print(f"  [INFO] Last 3 error lines:")
            for line in error_lines[-3:]:
                print(f"    {line.strip()[:100]}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to read log file: {e}")

def check_checkpoint_timesteps():
    """Check checkpoint files for timestep values"""
    print("\n[2] Checking Checkpoint Files for Timestep Values...")
    print("-" * 70)
    
    models_dir = project_root / "models"
    if not models_dir.exists():
        print(f"  [WARN] Models directory not found: {models_dir}")
        return
    
    # Look for checkpoint files
    checkpoint_files = list(models_dir.glob("checkpoint_*.pt"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not checkpoint_files:
        print(f"  [WARN] No checkpoint files found")
        return
    
    print(f"  [OK] Found {len(checkpoint_files)} checkpoint file(s)")
    
    # Check most recent checkpoint
    recent_checkpoint = checkpoint_files[0]
    print(f"  [INFO] Checking most recent checkpoint: {recent_checkpoint.name}")
    
    try:
        import torch
        checkpoint = torch.load(str(recent_checkpoint), map_location='cpu', weights_only=False)
        
        timestep = checkpoint.get("timestep", None)
        episode = checkpoint.get("episode", None)
        
        if timestep is not None:
            print(f"  [OK] Checkpoint timestep: {timestep:,}")
        else:
            print(f"  [WARN] No timestep in checkpoint")
        
        if episode is not None:
            print(f"  [OK] Checkpoint episode: {episode}")
        else:
            print(f"  [WARN] No episode in checkpoint")
        
        # Check if checkpoint is recent
        import time
        age_hours = (time.time() - recent_checkpoint.stat().st_mtime) / 3600
        print(f"  [INFO] Checkpoint age: {age_hours:.1f} hours")
        
    except Exception as e:
        print(f"  [ERROR] Failed to read checkpoint: {e}")

def check_training_code():
    """Check the training code for potential issues"""
    print("\n[3] Analyzing Training Code for Timestep Issues...")
    print("-" * 70)
    
    train_file = project_root / "src" / "train.py"
    if not train_file.exists():
        print(f"  [ERROR] Training file not found: {train_file}")
        return
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for timestep initialization
        if 'self.timestep = 0' in content:
            print(f"  [OK] Found timestep initialization")
            # Count occurrences
            count = content.count('self.timestep = 0')
            if count > 1:
                print(f"  [WARN] Timestep is reset to 0 in {count} places (might be resetting)")
        
        # Check for timestep increment
        if 'self.timestep += 1' in content:
            print(f"  [OK] Found timestep increment")
        else:
            print(f"  [ERROR] Timestep increment not found!")
        
        # Check if timestep increment is inside the training loop
        if 'while self.timestep < self.total_timesteps:' in content:
            print(f"  [OK] Training loop condition found")
            
            # Check if increment is after the loop starts
            loop_start = content.find('while self.timestep < self.total_timesteps:')
            increment_pos = content.find('self.timestep += 1')
            
            if increment_pos > loop_start:
                print(f"  [OK] Timestep increment is inside training loop")
            else:
                print(f"  [ERROR] Timestep increment is OUTSIDE training loop!")
        
        # Check for exceptions that might skip increment
        if 'except' in content and 'self.timestep' in content:
            print(f"  [INFO] Exception handling found - checking if timestep increment might be skipped")
            # This is complex to analyze statically, but we can note it
        
    except Exception as e:
        print(f"  [ERROR] Failed to analyze training code: {e}")

def check_api_response():
    """Try to get current timestep from API"""
    print("\n[4] Checking API Response for Current Timestep...")
    print("-" * 70)
    
    try:
        import requests
        response = requests.get("http://localhost:8000/api/training/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            timestep = metrics.get("timestep", 0)
            total_timesteps = metrics.get("total_timesteps", 0)
            episode = metrics.get("episode", 0)
            
            print(f"  [OK] API Response:")
            print(f"    Timestep: {timestep:,}")
            print(f"    Total timesteps: {total_timesteps:,}")
            print(f"    Episode: {episode}")
            print(f"    Progress: {(timestep/total_timesteps*100) if total_timesteps > 0 else 0:.2f}%")
            
            if timestep == 0 and episode > 0:
                print(f"\n  [ISSUE] ⚠️  Timestep is 0 but episodes are progressing!")
                print(f"         This indicates timestep is not being incremented properly")
                print(f"         Possible causes:")
                print(f"         1. Training loop is not executing (episodes completing immediately)")
                print(f"         2. Timestep increment is being skipped due to exceptions")
                print(f"         3. Trainer object is being recreated/reset")
                print(f"         4. Timestep increment is outside the loop or in wrong place")
        else:
            print(f"  [WARN] API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Cannot connect to API server")
    except Exception as e:
        print(f"  [ERROR] Failed to check API: {e}")

def main():
    """Run all diagnostics"""
    print("=" * 70)
    print("TIMESTEP DIAGNOSTIC")
    print("=" * 70)
    
    check_backend_logs()
    check_checkpoint_timesteps()
    check_training_code()
    check_api_response()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("\n  If timestep is 0 but episodes are progressing:")
    print("    1. Check backend console for exceptions during env.step()")
    print("    2. Verify training loop is actually running (not stuck)")
    print("    3. Check if trainer object is being recreated")
    print("    4. Verify timestep increment happens AFTER env.step()")
    print("\n  The timestep increment is at line 972 in src/train.py")
    print("  It should execute every iteration of the training loop")
    print("=" * 70)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("[ERROR] requests library not installed. Install with: pip install requests")
        sys.exit(1)
    
    main()

