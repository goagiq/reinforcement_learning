
"""
Quick Model Health and Performance Check

Checks:
1. Latest checkpoint info
2. Best model performance
3. Trade activity
4. Recommendations for optimization
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_evaluation import ModelEvaluator
from src.analyze_model import analyze_checkpoint

def check_checkpoints():
    """Check available checkpoints"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("ERROR: models directory not found!")
        return None, None
    
    # Find latest checkpoint
    checkpoints = sorted(
        models_dir.glob("checkpoint_*.pt"),
        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0
    )
    
    latest = checkpoints[-1] if checkpoints else None
    best = models_dir / "best_model.pt"
    
    return latest, best if best.exists() else None

def analyze_model_performance(model_path, n_episodes=3):
    """Quick performance check"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {Path(model_path).name}")
    print(f"{'='*70}")
    
    try:
        # Load config
        config_path = Path("configs/train_config_full.yaml")
        if not config_path.exists():
            print("ERROR: Config file not found!")
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Evaluate
        metrics = evaluator.evaluate_model(
            model_path=str(model_path),
            n_episodes=n_episodes,
            deterministic=True
        )
        
        return metrics
        
    except Exception as e:
        print(f"ERROR evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("MODEL HEALTH AND PERFORMANCE CHECK")
    print("="*70)
    
    # Check checkpoints
    print("\n[1] Checking Checkpoints...")
    latest, best = check_checkpoints()
    
    if not latest and not best:
        print("ERROR: No checkpoints found!")
        print("   Start training with: uv run python src/train.py")
        return
    
    if latest:
        latest_steps = int(latest.stem.split('_')[1])
        print(f"   Latest checkpoint: {latest.name} ({latest_steps:,} steps)")
    
    if best:
        print(f"   Best model: {best.name}")
    
    # Analyze checkpoint structure
    print("\n[2] Analyzing Checkpoint Structure...")
    model_to_check = best if best else latest
    if model_to_check:
        try:
            analyze_checkpoint(str(model_to_check))
        except Exception as e:
            print(f"   Warning: Could not analyze checkpoint: {e}")
    
    # Evaluate performance
    print("\n[3] Evaluating Model Performance...")
    print("   (This may take a minute...)")
    
    model_to_eval = best if best else latest
    if model_to_eval:
        metrics = analyze_model_performance(model_to_eval, n_episodes=5)
        
        if metrics:
            print(f"\n{'='*70}")
            print("PERFORMANCE METRICS")
            print(f"{'='*70}")
            print(f"Total Return:     {metrics.total_return*100:>8.2f}%")
            print(f"Sharpe Ratio:     {metrics.sharpe_ratio:>8.2f}")
            print(f"Sortino Ratio:   {metrics.sortino_ratio:>8.2f}")
            print(f"Win Rate:        {metrics.win_rate*100:>8.1f}%")
            print(f"Profit Factor:   {metrics.profit_factor:>8.2f}")
            print(f"Max Drawdown:    {metrics.max_drawdown*100:>8.2f}%")
            print(f"Total Trades:    {metrics.total_trades:>8}")
            print(f"Average Win:     ${metrics.average_win:>8.2f}")
            
            # Health assessment
            print(f"\n{'='*70}")
            print("HEALTH ASSESSMENT")
            print(f"{'='*70}")
            
            issues = []
            recommendations = []
            
            if metrics.total_trades == 0:
                issues.append("CRITICAL: No trades executed!")
                recommendations.append("  - Model may be too conservative")
                recommendations.append("  - Check action range and exploration")
                recommendations.append("  - Consider increasing entropy_coef")
                recommendations.append("  - Review reward function - may be too penalizing")
            elif metrics.total_trades < 5:
                issues.append("WARNING: Very few trades (< 5)")
                recommendations.append("  - Model is too conservative")
                recommendations.append("  - Increase entropy_coef for more exploration")
            
            if metrics.win_rate < 0.4:
                issues.append("WARNING: Low win rate (< 40%)")
                recommendations.append("  - Model may need more training")
                recommendations.append("  - Consider adjusting reward function")
            
            if metrics.total_return < 0:
                issues.append("WARNING: Negative total return")
                recommendations.append("  - Model is losing money")
                recommendations.append("  - May need parameter tuning")
                recommendations.append("  - Check if transaction costs are too high")
            
            if metrics.sharpe_ratio < 0.5:
                issues.append("WARNING: Low Sharpe ratio (< 0.5)")
                recommendations.append("  - High risk relative to returns")
                recommendations.append("  - Consider risk management improvements")
            
            if not issues:
                print("STATUS: Model appears healthy!")
                print("  - Trades are being executed")
                print("  - Performance metrics are reasonable")
            else:
                print("ISSUES DETECTED:")
                for issue in issues:
                    print(f"  - {issue}")
            
            if recommendations:
                print(f"\nRECOMMENDATIONS:")
                for rec in recommendations:
                    print(rec)
            
            # Parameter optimization suggestions
            print(f"\n{'='*70}")
            print("PARAMETER OPTIMIZATION SUGGESTIONS")
            print(f"{'='*70}")
            
            if metrics.total_trades == 0:
                print("\n1. Increase Exploration:")
                print("   - entropy_coef: 0.01 -> 0.05 (or higher)")
                print("   - action_range: [-1.0, 1.0] -> [-1.5, 1.5] (wider range)")
                print("   - clip_range: 0.2 -> 0.3 (allow more policy change)")
            
            if metrics.total_trades > 0 and metrics.win_rate < 0.45:
                print("\n2. Improve Trade Quality:")
                print("   - Review reward function weights")
                print("   - Consider adding technical indicator features")
                print("   - Check if stop-loss/take-profit logic needs adjustment")
            
            if metrics.total_return < 0 and metrics.total_trades > 10:
                print("\n3. Fix Profitability:")
                print("   - transaction_cost may be too high (currently 0.0001)")
                print("   - risk_penalty may be too high (currently 0.1)")
                print("   - Consider reducing drawdown_penalty")
            
            print(f"\n{'='*70}")
    
    # Training summary
    print("\n[4] Training Summary...")
    summary_path = Path("logs/training_summary.json")
    if summary_path.exists():
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print(f"   Total Timesteps: {summary.get('total_timesteps', 0):,}")
        print(f"   Total Episodes: {summary.get('total_episodes', 0)}")
        print(f"   Mean Reward: {summary.get('mean_reward', 0):.2f}")
        print(f"   Best Reward: {summary.get('best_reward', 0):.2f}")
        
        if summary.get('mean_reward', 0) < 0:
            print("   WARNING: Mean reward is negative!")
    
    print(f"\n{'='*70}")
    print("CHECK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

