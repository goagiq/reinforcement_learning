"""
Verify if Priority 1 features are active in the current training session
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import torch
from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor

def verify_priority1_features():
    """Verify Priority 1 features are enabled in the environment"""
    
    print("=" * 60)
    print("PRIORITY 1 FEATURES VERIFICATION")
    print("=" * 60)
    
    # Load config
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False
    
    print(f"\n[1] Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check config settings
    print(f"\n[2] Checking config settings...")
    slippage_config = config.get("environment", {}).get("reward", {}).get("slippage", {})
    market_impact_config = config.get("environment", {}).get("reward", {}).get("market_impact", {})
    
    slippage_enabled = slippage_config.get("enabled", False)
    market_impact_enabled = market_impact_config.get("enabled", False)
    
    print(f"  Slippage enabled in config: {slippage_enabled}")
    print(f"  Market impact enabled in config: {market_impact_enabled}")
    
    # Check module availability
    print(f"\n[3] Checking module availability...")
    try:
        from src.slippage_model import SlippageModel
        from src.market_impact import MarketImpactModel
        from src.execution_quality import ExecutionQualityTracker
        print("  [OK] All Priority 1 modules are available")
        modules_available = True
    except ImportError as e:
        print(f"  [ERROR] Module import error: {e}")
        modules_available = False
    
    # Verify code has the print statements
    print(f"\n[4] Checking code for Priority 1 initialization messages...")
    trading_env_file = Path("src/trading_env.py")
    if trading_env_file.exists():
        with open(trading_env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_slippage_msg = "Slippage model:" in content and "Enabled" in content
            has_market_impact_msg = "Market impact model:" in content and "Enabled" in content
            has_execution_tracker_msg = "Execution quality tracker:" in content
            
            print(f"  Code has slippage message: {has_slippage_msg}")
            print(f"  Code has market impact message: {has_market_impact_msg}")
            print(f"  Code has execution tracker message: {has_execution_tracker_msg}")
            
            if has_slippage_msg and has_market_impact_msg and has_execution_tracker_msg:
                print("  [OK] All Priority 1 messages are in the code")
            else:
                print("  [WARN] Some Priority 1 messages may be missing")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    
    if slippage_enabled and market_impact_enabled and modules_available:
        print("[SUCCESS] Priority 1 features should be ACTIVE!")
        print("=" * 60)
        print("  [OK] Config: Slippage enabled")
        print("  [OK] Config: Market impact enabled")
        print("  [OK] Modules: All Priority 1 modules available")
        print("\n[INFO] To verify messages appeared during training:")
        print("  1. Check the console/terminal where you started the backend")
        print("  2. Look for these messages when training started:")
        print("     'Creating trading environment...'")
        print("     '  [OK] Slippage model: Enabled'")
        print("     '  [OK] Market impact model: Enabled'")
        print("     '  [OK] Execution quality tracker: Available'")
        print("\n[NOTE] If training runs in a background thread, messages")
        print("  may not appear in the console. Check the API server console.")
        return True
    else:
        print("[WARNING] Priority 1 features may not be fully configured")
        print("=" * 60)
        if not slippage_enabled:
            print("  [ERROR] Slippage is disabled in config")
        if not market_impact_enabled:
            print("  [ERROR] Market impact is disabled in config")
        if not modules_available:
            print("  [ERROR] Priority 1 modules are not available")
        return False

if __name__ == "__main__":
    success = verify_priority1_features()
    sys.exit(0 if success else 1)

