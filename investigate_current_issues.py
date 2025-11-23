"""
Investigation Script for Current Training Issues
Checks data length, episode termination, and configuration
"""

import yaml
from pathlib import Path
from src.data_extraction import DataExtractor

def check_data_length():
    """Check if data is long enough for episodes"""
    print("=" * 80)
    print("1. DATA LENGTH CHECK")
    print("=" * 80)
    
    try:
        with open('configs/train_config_adaptive.yaml') as f:
            config = yaml.safe_load(f)
        
        extractor = DataExtractor(config)
        data = extractor.load_data()
        
        primary_data = data[min(config['environment']['timeframes'])]
        max_steps = config['environment']['max_episode_steps']
        lookback = config['environment']['lookback_bars']
        required = max_steps + lookback
        
        print(f"Data length: {len(primary_data):,}")
        print(f"Max episode steps: {max_steps:,}")
        print(f"Lookback bars: {lookback}")
        print(f"Required length: {required:,}")
        print(f"Margin: {len(primary_data) - required:,} bars")
        
        if len(primary_data) >= required:
            print("✅ Data is long enough")
        else:
            print(f"❌ Data is too short! Need {required:,}, have {len(primary_data):,}")
        
        return len(primary_data) >= required
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_max_consecutive_losses_fix():
    """Verify max_consecutive_losses fix is applied"""
    print("\n" + "=" * 80)
    print("2. MAX_CONSECUTIVE_LOSSES FIX VERIFICATION")
    print("=" * 80)
    
    try:
        with open('src/trading_env.py', 'r') as f:
            lines = f.readlines()
        
        # Find where max_consecutive_losses is defined
        definition_line = None
        first_use_line = None
        
        for i, line in enumerate(lines, 1):
            if 'max_consecutive_losses = self.reward_config.get' in line:
                definition_line = i
            if first_use_line is None and 'max_consecutive_losses' in line and '=' not in line:
                # This is likely a use, not definition
                if 'if' in line or '>=' in line or '<' in line:
                    first_use_line = i
        
        print(f"Definition found at line: {definition_line}")
        print(f"First use found at line: {first_use_line}")
        
        if definition_line and first_use_line:
            if definition_line < first_use_line:
                print("✅ Fix is applied - defined before first use")
                return True
            else:
                print("❌ Fix NOT applied - used before definition")
                return False
        else:
            print("⚠️  Could not verify - check manually")
            return None
    except Exception as e:
        print(f"❌ Error checking fix: {e}")
        return False

def check_adaptive_training():
    """Check adaptive training configuration"""
    print("\n" + "=" * 80)
    print("3. ADAPTIVE TRAINING STATUS")
    print("=" * 80)
    
    try:
        with open('configs/train_config_adaptive.yaml') as f:
            config = yaml.safe_load(f)
        
        adaptive = config.get('training', {}).get('adaptive_training', {})
        enabled = adaptive.get('enabled', False)
        
        print(f"Adaptive training enabled: {enabled}")
        
        if enabled:
            print(f"  Eval frequency: {adaptive.get('eval_frequency', 'N/A')}")
            print(f"  Eval episodes: {adaptive.get('eval_episodes', 'N/A')}")
            print(f"  Min trades/episode: {adaptive.get('min_trades_per_episode', 'N/A')}")
            print("✅ Adaptive training is ENABLED")
        else:
            print("⚠️  Adaptive training is DISABLED")
        
        return enabled
    except Exception as e:
        print(f"❌ Error checking adaptive training: {e}")
        return False

def check_quality_filters():
    """Check quality filter settings"""
    print("\n" + "=" * 80)
    print("4. QUALITY FILTER SETTINGS")
    print("=" * 80)
    
    try:
        with open('configs/train_config_adaptive.yaml') as f:
            config = yaml.safe_load(f)
        
        quality = config.get('environment', {}).get('reward', {}).get('quality_filters', {})
        decision_gate = config.get('decision_gate', {})
        
        print("Quality Filters:")
        print(f"  Enabled: {quality.get('enabled', False)}")
        print(f"  Min action confidence: {quality.get('min_action_confidence', 'N/A')}")
        print(f"  Min quality score: {quality.get('min_quality_score', 'N/A')}")
        
        print("\nDecisionGate:")
        print(f"  Min combined confidence: {decision_gate.get('min_combined_confidence', 'N/A')}")
        print(f"  Min confluence required: {decision_gate.get('min_confluence_required', 'N/A')}")
        
        print("\nAction Threshold:")
        print(f"  Action threshold: {config.get('environment', {}).get('action_threshold', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Error checking filters: {e}")
        return False

def check_stop_loss():
    """Check stop loss configuration"""
    print("\n" + "=" * 80)
    print("5. STOP LOSS CONFIGURATION")
    print("=" * 80)
    
    try:
        with open('configs/train_config_adaptive.yaml') as f:
            config = yaml.safe_load(f)
        
        stop_loss = config.get('environment', {}).get('reward', {}).get('stop_loss_pct', None)
        max_consecutive = config.get('environment', {}).get('reward', {}).get('max_consecutive_losses', None)
        
        print(f"Stop loss percentage: {stop_loss}")
        print(f"Max consecutive losses: {max_consecutive}")
        
        if stop_loss:
            print(f"✅ Stop loss configured at {stop_loss * 100}%")
        else:
            print("⚠️  Stop loss not configured")
        
        return stop_loss is not None
    except Exception as e:
        print(f"❌ Error checking stop loss: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "=" * 80)
    print("TRAINING ISSUES INVESTIGATION")
    print("=" * 80)
    print()
    
    results = {}
    results['data_length'] = check_data_length()
    results['max_consecutive_fix'] = check_max_consecutive_losses_fix()
    results['adaptive_training'] = check_adaptive_training()
    results['quality_filters'] = check_quality_filters()
    results['stop_loss'] = check_stop_loss()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for check, result in results.items():
        status = "✅" if result else "❌" if result is False else "⚠️"
        print(f"{status} {check}: {result}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Check backend logs for exception messages")
    print("2. Test episode termination in isolation")
    print("3. Review quality filter rejection reasons")
    print("4. Calculate average win vs loss sizes")

if __name__ == "__main__":
    main()

