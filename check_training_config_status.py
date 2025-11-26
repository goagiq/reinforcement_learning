"""
Quick script to check if current training is using Priority 1 features
(checks config file and verifies what would be loaded)
"""

import yaml
from pathlib import Path

def check_config_status():
    """Check config file for Priority 1 feature settings"""
    
    config_path = Path("configs/train_config_adaptive.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    print(f"[INFO] Checking config: {config_path}")
    print("=" * 60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check slippage settings
    slippage_config = config.get("environment", {}).get("reward", {}).get("slippage", {})
    slippage_enabled = slippage_config.get("enabled", False)
    
    # Check market impact settings
    market_impact_config = config.get("environment", {}).get("reward", {}).get("market_impact", {})
    market_impact_enabled = market_impact_config.get("enabled", False)
    
    print(f"\n[PRIORITY 1 FEATURES STATUS]")
    print(f"  Slippage: {'[ENABLED]' if slippage_enabled else '[DISABLED]'}")
    if slippage_enabled:
        print(f"    - Base slippage: {slippage_config.get('base_slippage', 'N/A')}")
        print(f"    - Volatility factor: {slippage_config.get('volatility_factor', 'N/A')}")
        print(f"    - Volume factor: {slippage_config.get('volume_factor', 'N/A')}")
    
    print(f"  Market Impact: {'[ENABLED]' if market_impact_enabled else '[DISABLED]'}")
    if market_impact_enabled:
        print(f"    - Impact coefficient: {market_impact_config.get('impact_coefficient', 'N/A')}")
        print(f"    - Volume threshold: {market_impact_config.get('volume_threshold', 'N/A')}")
    
    print(f"\n[SUMMARY]")
    if slippage_enabled and market_impact_enabled:
        print("  [OK] Both Priority 1 features are ENABLED in config")
        print("  [OK] If training started AFTER config update, features are ACTIVE")
        print("  [WARN] If training started BEFORE config update, restart needed")
    elif slippage_enabled or market_impact_enabled:
        print("  [WARN] Only one Priority 1 feature is enabled")
    else:
        print("  [ERROR] Priority 1 features are DISABLED in config")
    
    # Check if modules are available
    print(f"\n[MODULE AVAILABILITY]")
    try:
        from src.slippage_model import SlippageModel
        from src.market_impact import MarketImpactModel
        from src.execution_quality import ExecutionQualityTracker
        print("  [OK] All Priority 1 modules are available")
    except ImportError as e:
        print(f"  [ERROR] Module import error: {e}")
        print("  [WARN] Features will not work even if enabled in config")

if __name__ == "__main__":
    check_config_status()

