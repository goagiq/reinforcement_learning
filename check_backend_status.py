"""
Quick check to see if there are any obvious issues with the code changes
"""

import sys
from pathlib import Path

# Check if adaptive config file exists (this is OK if it doesn't)
adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
if adaptive_config_path.exists():
    print(f"[OK] Adaptive config file exists: {adaptive_config_path}")
    try:
        import json
        with open(adaptive_config_path, 'r') as f:
            config = json.load(f)
        print(f"[OK] Config file is valid JSON")
        print(f"   Keys: {list(config.keys())}")
    except Exception as e:
        print(f"[ERROR] Config file exists but is invalid: {e}")
else:
    print(f"[INFO] Adaptive config file does not exist (this is OK for first run)")
    print(f"   Path: {adaptive_config_path}")

# Check if config file is valid
config_path = Path("configs/train_config_adaptive.yaml")
if config_path.exists():
    print(f"\n[OK] Training config exists: {config_path}")
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Config file is valid YAML")
        
        # Check reward config
        reward_config = config.get("environment", {}).get("reward", {})
        print(f"\nReward Config Values:")
        print(f"  inaction_penalty: {reward_config.get('inaction_penalty', 'N/A')}")
        print(f"  max_consecutive_losses: {reward_config.get('max_consecutive_losses', 'N/A')}")
        print(f"  exploration_bonus_scale: {reward_config.get('exploration_bonus_scale', 'N/A')}")
        print(f"  loss_mitigation: {reward_config.get('loss_mitigation', 'N/A')}")
        
        quality_filters = reward_config.get("quality_filters", {})
        print(f"\nQuality Filters:")
        print(f"  enabled: {quality_filters.get('enabled', 'N/A')}")
        print(f"  min_action_confidence: {quality_filters.get('min_action_confidence', 'N/A')}")
        print(f"  min_quality_score: {quality_filters.get('min_quality_score', 'N/A')}")
        
        action_threshold = config.get("environment", {}).get("action_threshold", "N/A")
        print(f"\nAction Threshold: {action_threshold}")
        
    except Exception as e:
        print(f"[ERROR] Config file exists but is invalid: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[ERROR] Training config file not found: {config_path}")

print("\n" + "="*60)
print("If backend is stuck, check:")
print("1. Backend console logs for errors")
print("2. Data loading progress (may take several minutes with many files)")
print("3. Look for 'Loading data...' or 'Creating trading environment...' messages")
print("="*60)

