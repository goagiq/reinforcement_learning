"""
Fix Training Issues

1. Fix consecutive loss limit bug
2. Review and adjust reward function parameters
3. Investigate consecutive loss limit logic
"""

import yaml
from pathlib import Path

def fix_config():
    """Review and adjust reward function parameters"""
    
    print("="*60)
    print("FIXING TRAINING ISSUES")
    print("="*60)
    
    config_path = Path("configs/train_config_adaptive.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. CURRENT REWARD PARAMETERS")
    print("-" * 60)
    reward_config = config.get("environment", {}).get("reward", {})
    
    print(f"Inaction Penalty: {reward_config.get('inaction_penalty', 0.0001)}")
    print(f"Action Threshold: {config.get('environment', {}).get('action_threshold', 0.02)}")
    print(f"Min Action Confidence: {reward_config.get('quality_filters', {}).get('min_action_confidence', 0.15)}")
    print(f"Min Quality Score: {reward_config.get('quality_filters', {}).get('min_quality_score', 0.4)}")
    print(f"Max Consecutive Losses: {reward_config.get('max_consecutive_losses', 3)}")
    print(f"Exploration Bonus Scale: {reward_config.get('exploration_bonus_scale', 0.00001)}")
    print(f"Loss Mitigation: {reward_config.get('loss_mitigation', 0.05)}")
    
    print("\n2. RECOMMENDED ADJUSTMENTS")
    print("-" * 60)
    
    # Based on analysis:
    # - Latest episode: 180 steps (very short)
    # - Mean reward: -1.38 (negative)
    # - 0.30 trades/episode (very low)
    # - Win rate: 37.8% (close to breakeven)
    
    recommendations = {
        "inaction_penalty": 0.00005,  # Reduce by 50% (from 0.0001) - less punitive
        "action_threshold": 0.015,  # Reduce from 0.02 to 0.015 (1.5%) - allow more trades
        "min_action_confidence": 0.12,  # Reduce from 0.15 to 0.12 - allow more trades
        "min_quality_score": 0.35,  # Reduce from 0.4 to 0.35 - allow more trades
        "max_consecutive_losses": 5,  # Increase from 3 to 5 - less restrictive
        "exploration_bonus_scale": 0.00002,  # Increase from 0.00001 to 0.00002 - encourage more exploration
        "loss_mitigation": 0.08,  # Increase from 0.05 to 0.08 - reduce penalty for losses
    }
    
    print("Recommended changes to improve training:")
    for key, value in recommendations.items():
        if key == "action_threshold":
            current = config.get("environment", {}).get("action_threshold", 0.02)
        elif key in ["min_action_confidence", "min_quality_score"]:
            current = reward_config.get("quality_filters", {}).get(key, 0.15 if "confidence" in key else 0.4)
        else:
            current = reward_config.get(key, 0.0001 if key == "inaction_penalty" else (3 if key == "max_consecutive_losses" else 0.00001))
        
        change_pct = ((value - current) / current * 100) if current > 0 else 0
        print(f"  {key}: {current} -> {value} ({change_pct:+.1f}%)")
    
    print("\n3. APPLYING FIXES")
    print("-" * 60)
    
    # Apply fixes
    if "environment" not in config:
        config["environment"] = {}
    if "reward" not in config["environment"]:
        config["environment"]["reward"] = {}
    if "quality_filters" not in config["environment"]["reward"]:
        config["environment"]["reward"]["quality_filters"] = {}
    
    # Update reward parameters
    config["environment"]["reward"]["inaction_penalty"] = recommendations["inaction_penalty"]
    config["environment"]["reward"]["max_consecutive_losses"] = recommendations["max_consecutive_losses"]
    config["environment"]["reward"]["exploration_bonus_scale"] = recommendations["exploration_bonus_scale"]
    config["environment"]["reward"]["loss_mitigation"] = recommendations["loss_mitigation"]
    
    # Update action threshold
    config["environment"]["action_threshold"] = recommendations["action_threshold"]
    
    # Update quality filters
    config["environment"]["reward"]["quality_filters"]["min_action_confidence"] = recommendations["min_action_confidence"]
    config["environment"]["reward"]["quality_filters"]["min_quality_score"] = recommendations["min_quality_score"]
    
    # Save updated config
    backup_path = config_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"[OK] Created backup: {backup_path}")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Updated config: {config_path}")
    print("\n4. SUMMARY OF CHANGES")
    print("-" * 60)
    print("Changes made to improve training:")
    print("  - Reduced inaction penalty (less punitive for not trading)")
    print("  - Lowered action threshold (allow more trades)")
    print("  - Relaxed quality filters (allow more exploration)")
    print("  - Increased consecutive loss limit (less restrictive)")
    print("  - Increased exploration bonus (encourage trading)")
    print("  - Increased loss mitigation (reduce penalty for losses)")
    print("\nThese changes should:")
    print("  - Increase trade count per episode")
    print("  - Improve reward signals (less negative)")
    print("  - Allow episodes to complete fully")
    print("  - Enable better learning through more exploration")
    
    print("\n" + "="*60)
    print("FIXES APPLIED")
    print("="*60)


if __name__ == "__main__":
    fix_config()

