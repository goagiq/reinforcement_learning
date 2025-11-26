"""
Test Supervised Pre-training Module

Quick test to verify pre-training works before full integration.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import torch
from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.trading_hours import TradingHoursManager
from src.supervised_pretraining import SupervisedPretrainer
from src.models import ActorNetwork
from src.utils.colors import success, info, warn, error

def test_pretraining():
    """Test supervised pre-training module"""
    
    print(info("\n" + "="*70))
    print(info("TESTING SUPERVISED PRE-TRAINING"))
    print(info("="*70))
    
    # Load config
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print(error(f"[TEST] Config file not found: {config_path}"))
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable pre-training in config
    if "pretraining" not in config:
        config["pretraining"] = {}
    config["pretraining"]["enabled"] = True
    
    print(info(f"[TEST] Loaded config from: {config_path}"))
    
    # Initialize data extractor
    print(info("\n[TEST] Initializing data extractor..."))
    data_extractor = DataExtractor()
    
    # Load data
    instrument = config["environment"]["instrument"]
    requested_timeframes = config["environment"]["timeframes"]
    
    print(info(f"[TEST] Loading data for {instrument} at timeframes {requested_timeframes}..."))
    
    trading_hours_config = config.get("environment", {}).get("trading_hours", {})
    if trading_hours_config:
        trading_hours = TradingHoursManager.from_dict(trading_hours_config)
    else:
        trading_hours = None
    
    data = {}
    for tf in requested_timeframes:
        try:
            df = data_extractor.load_historical_data(
                instrument=instrument,
                timeframe=tf,
                trading_hours=trading_hours
            )
            if df is not None and len(df) > 0:
                data[tf] = df
                print(success(f"  [OK] Loaded {len(df)} bars for {tf}min"))
            else:
                print(warn(f"  [WARN] No data loaded for {tf}min"))
        except Exception as e:
            print(warn(f"  [WARN] Failed to load {tf}min data: {e}"))
            # Continue with other timeframes
            continue
    
    if not data:
        print(error("[TEST] No data loaded!"))
        return False
    
    # Use only successfully loaded timeframes
    available_timeframes = list(data.keys())
    if not available_timeframes:
        print(error("[TEST] No data loaded for any timeframe!"))
        return False
    
    print(info(f"[TEST] Using available timeframes: {available_timeframes}"))
    
    # Create environment
    print(info("\n[TEST] Creating trading environment..."))
    try:
        env = TradingEnvironment(
            data=data,
            timeframes=available_timeframes,
            initial_capital=config["environment"].get("initial_capital", 100000.0),
            transaction_cost=config["environment"].get("transaction_cost", 0.0003),
            lookback_bars=config["environment"].get("lookback_bars", 20),
            reward_config=config.get("reward", {}),
            max_episode_steps=config["environment"].get("max_episode_steps"),
            action_threshold=config["environment"].get("action_threshold", 0.05)
        )
        print(success(f"  [OK] Environment created. State dim: {env.state_dim}"))
    except Exception as e:
        print(error(f"  [ERROR] Failed to create environment: {e}"))
        import traceback
        traceback.print_exc()
        return False
    
    # Create actor network
    print(info("\n[TEST] Creating actor network..."))
    try:
        state_dim = env.state_dim
        action_range = tuple(config["environment"]["action_range"])
        hidden_dims = config["model"]["hidden_dims"]
        
        actor = ActorNetwork(
            state_dim=state_dim,
            action_range=action_range,
            hidden_dims=hidden_dims
        )
        print(success(f"  [OK] Actor network created: {state_dim} -> {action_range}"))
    except Exception as e:
        print(error(f"  [ERROR] Failed to create actor network: {e}"))
        import traceback
        traceback.print_exc()
        return False
    
    # Test pre-training
    print(info("\n[TEST] Testing supervised pre-training..."))
    try:
        pretrainer = SupervisedPretrainer(
            config=config,
            data_extractor=data_extractor,
            device="cpu"  # Use CPU for testing
        )
        
        metrics = pretrainer.run_pretraining(
            actor=actor,
            env=env
        )
        
        if metrics:
            print(success("\n[TEST] Pre-training test PASSED!"))
            print(info(f"  Best validation loss: {metrics.get('best_val_loss', 'N/A')}"))
            print(info(f"  Epochs trained: {metrics.get('epochs_trained', 'N/A')}"))
            return True
        else:
            print(warn("[TEST] Pre-training returned no metrics (may be disabled)"))
            return False
            
    except Exception as e:
        print(error(f"[TEST] Pre-training test FAILED: {e}"))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success_flag = test_pretraining()
    sys.exit(0 if success_flag else 1)

