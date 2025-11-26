"""
Analyze state dimension mismatch issue
"""
import yaml
from pathlib import Path

def analyze_state_dimension():
    config_path = Path("configs/train_config_adaptive.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("STATE DIMENSION ANALYSIS")
    print("=" * 80)
    print()
    
    # Get config values
    state_features = config.get('environment', {}).get('state_features', 900)
    timeframes = config.get('environment', {}).get('timeframes', [1, 5, 15])
    lookback_bars = config.get('environment', {}).get('lookback_bars', 20)
    
    reward_config = config.get('environment', {}).get('reward', {})
    include_regime = reward_config.get('include_regime_features', False)
    include_forecast = reward_config.get('include_forecast_features', False)
    
    features_config = config.get('environment', {}).get('features', {})
    include_forecast_features = features_config.get('include_forecast_features', False)
    
    print("[CONFIG] Current Settings:")
    print(f"   state_features: {state_features}")
    print(f"   timeframes: {timeframes}")
    print(f"   lookback_bars: {lookback_bars}")
    print(f"   reward.include_regime_features: {include_regime}")
    print(f"   reward.include_forecast_features: {include_forecast}")
    print(f"   features.include_forecast_features: {include_forecast_features}")
    print()
    
    # Calculate expected state dimension
    # Base: 15 features per timeframe * 3 timeframes * 20 lookback = 900
    features_per_tf = 15  # OHLCV (5) + volume_ratio (1) + returns (1) + indicators (8)
    base_state_dim = features_per_tf * len(timeframes) * lookback_bars
    
    # Add regime features if enabled (5 features)
    regime_features_dim = 5 if include_regime else 0
    
    # Add forecast features if enabled (3 features)
    # Check both locations
    forecast_enabled = include_forecast or include_forecast_features
    forecast_features_dim = 3 if forecast_enabled else 0
    
    expected_state_dim = base_state_dim + regime_features_dim + forecast_features_dim
    
    print("[CALCULATION] Expected State Dimension:")
    print(f"   Base state dim: {base_state_dim} (15 features * {len(timeframes)} timeframes * {lookback_bars} lookback)")
    print(f"   Regime features: +{regime_features_dim} ({'enabled' if include_regime else 'disabled'})")
    print(f"   Forecast features: +{forecast_features_dim} ({'enabled' if forecast_enabled else 'disabled'})")
    print(f"   Expected total: {expected_state_dim}")
    print()
    
    # Check mismatch
    if state_features != expected_state_dim:
        print("[MISMATCH] State Dimension Mismatch Detected!")
        print(f"   Config state_features: {state_features}")
        print(f"   Expected state_dim: {expected_state_dim}")
        print(f"   Difference: {expected_state_dim - state_features}")
        print()
        print("[ISSUE] The config's state_features doesn't match the calculated state_dim")
        print("   This causes architecture mismatch when resuming from checkpoint")
        print()
        print("[RECOMMENDATION] Update config:")
        print(f"   Change 'state_features: {state_features}' to 'state_features: {expected_state_dim}'")
    else:
        print("[OK] State dimension matches expected value")
    
    print()
    print("=" * 80)
    print("PERFORMANCE IMPACT")
    print("=" * 80)
    print()
    print("When resuming from checkpoint with architecture mismatch:")
    print("1. Transfer learning is used (copy_and_extend strategy)")
    print("2. Existing weights are preserved for matching dimensions")
    print("3. New dimensions (regime + forecast) are initialized randomly")
    print("4. This can cause poor initial performance until model adapts")
    print()
    print("Current performance since checkpoint resume:")
    print("   - 121 trades")
    print("   - Total PnL: -$3,490.59")
    print("   - Win Rate: 46.28%")
    print("   - Profit Factor: 0.58 (target: >1.0)")
    print("   - Sharpe Ratio: -3.80 (target: >1.0)")
    print()
    print("[RECOMMENDATION]")
    print("1. Fix state_features in config to match expected dimension")
    print("2. Consider starting fresh training if performance doesn't improve")
    print("3. Or disable forecast/regime features to match checkpoint architecture")

if __name__ == "__main__":
    analyze_state_dimension()

