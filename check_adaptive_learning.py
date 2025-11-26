"""
Check adaptive learning status and verify if it's working correctly.

This script checks:
1. If adaptive learning is enabled and active
2. When the last adjustment was made
3. Current training timestep
4. Conditions that might prevent adjustments
5. Whether adjustments should be happening
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_adaptive_config():
    """Check if adaptive config file exists and read current parameters"""
    print("\n[1] Checking Adaptive Learning Configuration...")
    print("-" * 70)
    
    config_path = project_root / "logs/adaptive_training/current_reward_config.json"
    
    if not config_path.exists():
        print(f"  [WARN] Adaptive config file not found: {config_path}")
        print(f"         This suggests adaptive learning hasn't been initialized yet.")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"  [OK] Config file exists: {config_path}")
        print(f"\n  Current Parameters:")
        if "min_risk_reward_ratio" in config:
            print(f"    - R:R Ratio: {config['min_risk_reward_ratio']}")
        if "min_action_confidence" in config:
            print(f"    - Min Confidence: {config['min_action_confidence']}")
        if "min_quality_score" in config:
            print(f"    - Min Quality: {config['min_quality_score']}")
        if "entropy_coef" in config:
            print(f"    - Entropy Coef: {config['entropy_coef']}")
        if "inaction_penalty" in config:
            print(f"    - Inaction Penalty: {config['inaction_penalty']}")
        
        return config
    except Exception as e:
        print(f"  [ERROR] Failed to read config: {e}")
        return None

def check_adjustment_history():
    """Check adjustment history to see when last adjustment was made"""
    print("\n[2] Checking Adjustment History...")
    print("-" * 70)
    
    history_path = project_root / "logs/adaptive_training/config_adjustments.jsonl"
    
    if not history_path.exists():
        print(f"  [WARN] Adjustment history file not found: {history_path}")
        print(f"         No adjustments have been made yet.")
        return None, 0
    
    try:
        with open(history_path, 'r') as f:
            lines = f.readlines()
        
        total_adjustments = len([l for l in lines if l.strip()])
        print(f"  [OK] History file exists: {history_path}")
        print(f"  [OK] Total adjustments: {total_adjustments}")
        
        if total_adjustments == 0:
            print(f"  [WARN] No adjustments recorded yet.")
            return None, 0
        
        # Get last adjustment
        last_line = None
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break
        
        if last_line:
            last_adjustment = json.loads(last_line)
            timestamp_str = last_adjustment.get("timestamp", "")
            timestep = last_adjustment.get("timestep", 0)
            adjustments = last_adjustment.get("adjustments", {})
            
            print(f"\n  Last Adjustment:")
            print(f"    - Timestep: {timestep:,}")
            
            if timestamp_str:
                try:
                    # Parse ISO timestamp
                    if 'T' in timestamp_str:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromisoformat(timestamp_str)
                    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
                    time_diff = now - dt.replace(tzinfo=None) if dt.tzinfo else now - dt
                    hours_ago = time_diff.total_seconds() / 3600
                    print(f"    - Time: {timestamp_str}")
                    print(f"    - Time ago: {hours_ago:.1f} hours")
                except Exception as e:
                    print(f"    - Time: {timestamp_str} (parse error: {e})")
            
            if adjustments:
                print(f"    - Adjustments made:")
                for key, value in adjustments.items():
                    if isinstance(value, dict):
                        old_val = value.get("old", "?")
                        new_val = value.get("new", "?")
                        reason = value.get("reason", "")
                        print(f"      • {key}: {old_val} -> {new_val}")
                        if reason:
                            print(f"        Reason: {reason[:80]}")
                    else:
                        print(f"      • {key}: {value}")
            
            return last_adjustment, total_adjustments
        else:
            print(f"  [WARN] Could not parse last adjustment")
            return None, total_adjustments
            
    except Exception as e:
        print(f"  [ERROR] Failed to read adjustment history: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def check_training_status():
    """Check if training is running and get current timestep"""
    print("\n[3] Checking Training Status...")
    print("-" * 70)
    
    try:
        import requests
        # Try multiple ports in case API is on different port
        api_urls = [
            "http://localhost:8000/api/training/status",
            "http://127.0.0.1:8000/api/training/status",
        ]
        
        response = None
        for url in api_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    break
            except:
                continue
        
        if response and response.status_code == 200:
            data = response.json()
            
            status = data.get("status", "unknown")
            metrics = data.get("metrics", {})
            timestep = metrics.get("timestep", 0)
            total_timesteps = metrics.get("total_timesteps", 0)
            episode = metrics.get("episode", 0)
            
            print(f"  [OK] Training status: {status}")
            print(f"  [OK] Current timestep: {timestep:,}")
            print(f"  [OK] Total timesteps: {total_timesteps:,}")
            print(f"  [OK] Current episode: {episode}")
            
            if status == "running":
                progress = (timestep / total_timesteps * 100) if total_timesteps > 0 else 0
                print(f"  [OK] Progress: {progress:.1f}%")
                return timestep, episode, True
            else:
                print(f"  [WARN] Training is not running (status: {status})")
                return timestep, episode, False
        else:
            # Try to check checkpoint files as fallback
            print(f"  [WARN] API not accessible, checking checkpoint files...")
            return check_training_from_checkpoint()
    except requests.exceptions.ConnectionError:
        print(f"  [WARN] Cannot connect to API server, checking checkpoint files...")
        return check_training_from_checkpoint()
    except Exception as e:
        print(f"  [WARN] API check failed: {e}, checking checkpoint files...")
        return check_training_from_checkpoint()

def check_training_from_checkpoint():
    """Check training status from checkpoint files as fallback"""
    try:
        import torch
        checkpoint_path = project_root / "models" / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            timestep = checkpoint.get("timestep", 0)
            episode = checkpoint.get("episode", 0)
            
            print(f"  [OK] Found checkpoint: {checkpoint_path}")
            print(f"  [OK] Checkpoint timestep: {timestep:,}")
            print(f"  [OK] Checkpoint episode: {episode}")
            print(f"  [INFO] Note: This is from saved checkpoint, may not reflect current training")
            
            # Check if checkpoint was recently updated
            import os
            import time
            mtime = os.path.getmtime(checkpoint_path)
            age_hours = (time.time() - mtime) / 3600
            print(f"  [INFO] Checkpoint age: {age_hours:.1f} hours")
            
            if age_hours < 1.0:
                print(f"  [OK] Checkpoint is recent - training likely active")
                return timestep, episode, True
            else:
                print(f"  [WARN] Checkpoint is old - training may have stopped")
                return timestep, episode, False
        else:
            print(f"  [WARN] No checkpoint file found")
            return 0, 0, False
    except Exception as e:
        print(f"  [ERROR] Failed to read checkpoint: {e}")
        return 0, 0, False

def analyze_adjustment_conditions(last_adjustment, current_timestep, is_training):
    """Analyze why adjustments might not be happening"""
    print("\n[4] Analyzing Adjustment Conditions...")
    print("-" * 70)
    
    if not is_training:
        print(f"  [WARN] Training is not running - adjustments only happen during training")
        return
    
    if not last_adjustment:
        print(f"  [INFO] No previous adjustments found - first adjustment should happen soon")
        print(f"         Expected: Every 5,000 timesteps (evaluation) or when conditions are met")
        return
    
    last_timestep = last_adjustment.get("timestep", 0)
    timesteps_since_adjustment = current_timestep - last_timestep
    
    print(f"  Current timestep: {current_timestep:,}")
    print(f"  Last adjustment timestep: {last_timestep:,}")
    print(f"  Timesteps since last adjustment: {timesteps_since_adjustment:,}")
    
    # Check evaluation frequency (default: 5000)
    eval_frequency = 5000
    timesteps_until_next_eval = eval_frequency - (timesteps_since_adjustment % eval_frequency)
    
    print(f"\n  Evaluation Frequency: Every {eval_frequency:,} timesteps")
    print(f"  Timesteps until next evaluation: {timesteps_until_next_eval:,}")
    
    # Check minimum adjustment interval (1000 timesteps)
    min_interval = 1000
    if timesteps_since_adjustment < min_interval:
        print(f"\n  [INFO] Minimum adjustment interval ({min_interval:,} timesteps) not met yet")
        print(f"         Adjustments are rate-limited to prevent too frequent changes")
    else:
        print(f"\n  [OK] Minimum adjustment interval ({min_interval:,} timesteps) has passed")
    
    # Check if we should have had an evaluation
    if timesteps_since_adjustment >= eval_frequency:
        print(f"\n  [WARN] ⚠️  Should have had an evaluation by now!")
        print(f"         Expected evaluation every {eval_frequency:,} timesteps")
        print(f"         Possible reasons:")
        print(f"         1. Training just started (need at least {eval_frequency:,} timesteps)")
        print(f"         2. Adaptive trainer not being called during training")
        print(f"         3. Evaluation conditions not met (need sufficient episodes/data)")
    else:
        print(f"\n  [INFO] Next evaluation expected in {timesteps_until_next_eval:,} timesteps")
    
    # Check quick adjustment conditions
    print(f"\n  Quick Adjustment Conditions:")
    print(f"    - Requires: At least 10 episodes with PnL data")
    print(f"    - Frequency: Checked every episode")
    print(f"    - Trigger: Negative mean PnL in last 10 episodes")

def check_training_config():
    """Check if adaptive learning is enabled in training config"""
    print("\n[5] Checking Training Configuration...")
    print("-" * 70)
    
    config_path = project_root / "configs" / "train_config_adaptive.yaml"
    
    if not config_path.exists():
        print(f"  [WARN] Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for adaptive_training (correct key) or adaptive_learning (old key)
        training_config = config.get("training", {})
        adaptive_config = training_config.get("adaptive_training", {}) or config.get("adaptive_learning", {})
        enabled = adaptive_config.get("enabled", False)
        
        print(f"  [OK] Config file: {config_path}")
        print(f"  Adaptive Training Enabled: {enabled}")
        
        if enabled:
            eval_frequency = adaptive_config.get("eval_frequency", 5000)
            eval_episodes = adaptive_config.get("eval_episodes", 3)
            print(f"  Evaluation Frequency: Every {eval_frequency:,} timesteps")
            print(f"  Evaluation Episodes: {eval_episodes}")
        else:
            print(f"  [WARN] Adaptive training is DISABLED in config!")
            print(f"         Enable it by setting training.adaptive_training.enabled: true")
        
        return enabled
        
    except Exception as e:
        print(f"  [ERROR] Failed to read config: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 70)
    print("ADAPTIVE LEARNING STATUS CHECK")
    print("=" * 70)
    
    # Check config file
    config = check_adaptive_config()
    
    # Check adjustment history
    last_adjustment, total_adjustments = check_adjustment_history()
    
    # Check training status
    current_timestep, current_episode, is_training = check_training_status()
    
    # Check training config
    adaptive_enabled = check_training_config()
    
    # Analyze conditions
    analyze_adjustment_conditions(last_adjustment, current_timestep, is_training)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not adaptive_enabled:
        print("  [ISSUE] Adaptive learning is DISABLED in training config")
        print("         Solution: Enable it in configs/train_config_adaptive.yaml")
    elif not is_training:
        print("  [ISSUE] Training is not currently running")
        print("         Solution: Start training to enable adaptive adjustments")
    elif total_adjustments == 0:
        print("  [INFO] No adjustments made yet - this is normal if training just started")
        print("         Adjustments will begin after:")
        print("         - At least 5,000 timesteps (for evaluation-based adjustments)")
        print("         - At least 10 episodes with PnL data (for quick adjustments)")
    elif last_adjustment:
        last_timestep = last_adjustment.get("timestep", 0)
        timesteps_since = current_timestep - last_timestep
        if timesteps_since < 5000:
            print(f"  [OK] Last adjustment was recent ({timesteps_since:,} timesteps ago)")
            print(f"       Next evaluation expected in {5000 - timesteps_since:,} timesteps")
        else:
            print(f"  [WARN] Last adjustment was {timesteps_since:,} timesteps ago")
            print(f"         Should have had evaluation by now (every 5,000 timesteps)")
            print(f"         Possible issue: Adaptive trainer not being called during training")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("[ERROR] requests library not installed. Install with: pip install requests")
        sys.exit(1)
    
    main()

