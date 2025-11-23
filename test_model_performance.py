"""
Test Fine-Tuned Model Performance and Progress

This script evaluates the fine-tuned model's performance by:
1. Testing the latest checkpoint
2. Comparing with earlier checkpoints to show progress
3. Evaluating the best model
4. Generating a performance report
"""

import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_evaluation import ModelEvaluator, ModelMetrics
from src.analyze_model import analyze_checkpoint


def find_checkpoints(models_dir: str = "models", milestone_steps: Optional[List[int]] = None) -> Dict[str, str]:
    """
    Find checkpoints at specific milestones.
    
    Args:
        models_dir: Directory containing models
        milestone_steps: List of step counts to find (e.g., [100000, 1000000, 2000000])
    
    Returns:
        Dictionary mapping step counts to checkpoint paths
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return {}
    
    checkpoints = {}
    
    # Find latest checkpoint
    all_checkpoints = sorted(
        models_path.glob("checkpoint_*.pt"),
        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0
    )
    
    if all_checkpoints:
        latest = all_checkpoints[-1]
        latest_steps = int(latest.stem.split('_')[1])
        checkpoints['latest'] = str(latest)
        checkpoints[f'{latest_steps}'] = str(latest)
        print(f"✅ Found latest checkpoint: {latest.name} ({latest_steps:,} steps)")
    
    # Find best model
    best_model = models_path / "best_model.pt"
    if best_model.exists():
        checkpoints['best'] = str(best_model)
        print(f"✅ Found best model: {best_model.name}")
    
    # Find milestone checkpoints
    if milestone_steps:
        for milestone in milestone_steps:
            # Find closest checkpoint to milestone
            milestone_checkpoint = models_path / f"checkpoint_{milestone}.pt"
            if milestone_checkpoint.exists():
                checkpoints[f'{milestone}'] = str(milestone_checkpoint)
                print(f"✅ Found milestone checkpoint: {milestone_checkpoint.name}")
            else:
                # Find closest checkpoint
                closest = None
                min_diff = float('inf')
                for cp in all_checkpoints:
                    steps = int(cp.stem.split('_')[1])
                    diff = abs(steps - milestone)
                    if diff < min_diff and steps <= milestone + 10000:  # Within 10k steps
                        min_diff = diff
                        closest = cp
                
                if closest:
                    checkpoints[f'{milestone}'] = str(closest)
                    actual_steps = int(closest.stem.split('_')[1])
                    print(f"✅ Found closest to {milestone:,}: {closest.name} ({actual_steps:,} steps)")
    
    return checkpoints


def print_metrics_table(metrics_list: List[tuple], title: str = "Performance Comparison"):
    """Print a formatted comparison table"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    
    # Header
    print(f"{'Model':<30} {'Return %':<12} {'Sharpe':<10} {'Sortino':<10} {'Win Rate %':<12} {'Profit Factor':<12} {'Max DD %':<12} {'Trades':<10}")
    print("-"*80)
    
    # Data rows
    for name, metrics in metrics_list:
        name_short = Path(name).name if len(name) > 28 else name
        print(f"{name_short:<30} "
              f"{metrics.total_return*100:>10.2f}%  "
              f"{metrics.sharpe_ratio:>8.2f}  "
              f"{metrics.sortino_ratio:>8.2f}  "
              f"{metrics.win_rate*100:>10.1f}%  "
              f"{metrics.profit_factor:>10.2f}  "
              f"{metrics.max_drawdown*100:>10.2f}%  "
              f"{metrics.total_trades:>8}")
    
    print("="*80)


def print_progress_analysis(checkpoints_dict: Dict[str, str], results: Dict[str, ModelMetrics]):
    """Analyze and print progress over time"""
    print("\n" + "="*80)
    print("  PROGRESS ANALYSIS")
    print("="*80)
    
    # Sort by step count
    sorted_results = []
    for key, metrics in results.items():
        if key.isdigit():
            sorted_results.append((int(key), metrics))
    
    sorted_results.sort()
    
    if len(sorted_results) < 2:
        print("⚠️  Need at least 2 checkpoints to show progress")
        return
    
    print("\n📈 Performance Trends:")
    print(f"{'Checkpoint':<20} {'Return %':<15} {'Sharpe':<15} {'Win Rate %':<15} {'Improvement'}")
    print("-"*80)
    
    prev_sharpe = None
    prev_return = None
    
    for steps, metrics in sorted_results:
        improvement = ""
        if prev_sharpe is not None:
            sharpe_change = metrics.sharpe_ratio - prev_sharpe
            return_change = (metrics.total_return - prev_return) * 100
            improvement = f"Sharpe: {sharpe_change:+.2f}, Return: {return_change:+.2f}%"
        
        print(f"{steps:>15,} steps  "
              f"{metrics.total_return*100:>12.2f}%  "
              f"{metrics.sharpe_ratio:>12.2f}  "
              f"{metrics.win_rate*100:>12.1f}%  "
              f"{improvement}")
        
        prev_sharpe = metrics.sharpe_ratio
        prev_return = metrics.total_return
    
    # Calculate overall improvement
    if len(sorted_results) >= 2:
        first = sorted_results[0][1]
        last = sorted_results[-1][1]
        
        sharpe_improvement = last.sharpe_ratio - first.sharpe_ratio
        return_improvement = (last.total_return - first.total_return) * 100
        
        print("\n" + "-"*80)
        print(f"📊 Overall Progress:")
        print(f"   Sharpe Ratio: {first.sharpe_ratio:.2f} → {last.sharpe_ratio:.2f} ({sharpe_improvement:+.2f})")
        print(f"   Total Return: {first.total_return*100:.2f}% → {last.total_return*100:.2f}% ({return_improvement:+.2f}%)")
        print(f"   Win Rate: {first.win_rate*100:.1f}% → {last.win_rate*100:.1f}% ({last.win_rate*100 - first.win_rate*100:+.1f}%)")


def save_report(results: Dict[str, ModelMetrics], output_file: str = "logs/model_performance_report.json"):
    """Save evaluation results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    for name, metrics in results.items():
        report["models"][name] = {
            "model_path": metrics.model_path,
            "timestamp": metrics.timestamp,
            "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_trades": metrics.total_trades,
            "average_win": metrics.average_win,
            "average_loss": metrics.average_loss
        }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Performance report saved to: {output_path}")


def main():
    """Main evaluation function"""
    print("="*80)
    print("  FINE-TUNED MODEL PERFORMANCE TEST")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    config_path = Path("configs/train_config_full.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ Loaded configuration")
    
    # Find checkpoints to evaluate
    # Evaluate at milestones: 1M, 2M, 3M, and latest
    milestone_steps = [1000000, 2000000, 3000000]
    checkpoints = find_checkpoints(milestone_steps=milestone_steps)
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    # Initialize evaluator
    print("\n📊 Initializing Model Evaluator...")
    evaluator = ModelEvaluator(config)
    
    # Evaluate models
    print("\n" + "="*80)
    print("  EVALUATING MODELS")
    print("="*80)
    
    results = {}
    n_episodes = 10  # Number of evaluation episodes
    
    # Evaluate each checkpoint
    checkpoint_items = sorted(
        [(k, v) for k, v in checkpoints.items() if k != 'best'],
        key=lambda x: int(x[0]) if x[0].isdigit() else 0
    )
    
    # Add best model at the end
    if 'best' in checkpoints:
        checkpoint_items.append(('best', checkpoints['best']))
    
    for name, checkpoint_path in checkpoint_items:
        print(f"\n{'='*80}")
        print(f"Evaluating: {Path(checkpoint_path).name}")
        print(f"{'='*80}")
        
        # Analyze checkpoint structure first
        try:
            analyze_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"⚠️  Could not analyze checkpoint structure: {e}")
        
        # Evaluate performance
        try:
            metrics = evaluator.evaluate_model(
                model_path=checkpoint_path,
                n_episodes=n_episodes,
                deterministic=True
            )
            results[name] = metrics
            
            # Print quick summary
            print(f"\n📊 Quick Summary:")
            print(f"   Total Return: {metrics.total_return*100:.2f}%")
            print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"   Win Rate: {metrics.win_rate*100:.1f}%")
            print(f"   Profit Factor: {metrics.profit_factor:.2f}")
            print(f"   Max Drawdown: {metrics.max_drawdown*100:.2f}%")
            print(f"   Total Trades: {metrics.total_trades}")
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    if results:
        # Prepare data for table
        metrics_list = []
        for name, metrics in results.items():
            display_name = f"{name} ({Path(metrics.model_path).name})"
            metrics_list.append((display_name, metrics))
        
        print_metrics_table(metrics_list, "Performance Comparison")
        
        # Print progress analysis
        print_progress_analysis(checkpoints, results)
        
        # Find best performing model
        best_by_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        best_by_return = max(results.items(), key=lambda x: x[1].total_return)
        
        print("\n" + "="*80)
        print("  BEST PERFORMING MODELS")
        print("="*80)
        print(f"🏆 Best by Sharpe Ratio: {best_by_sharpe[0]}")
        print(f"   Sharpe: {best_by_sharpe[1].sharpe_ratio:.2f}")
        print(f"   Return: {best_by_sharpe[1].total_return*100:.2f}%")
        print(f"   Model: {Path(best_by_sharpe[1].model_path).name}")
        
        print(f"\n💰 Best by Total Return: {best_by_return[0]}")
        print(f"   Return: {best_by_return[1].total_return*100:.2f}%")
        print(f"   Sharpe: {best_by_return[1].sharpe_ratio:.2f}")
        print(f"   Model: {Path(best_by_return[1].model_path).name}")
        
        # Save report
        save_report(results)
        
        print("\n" + "="*80)
        print("  ✅ EVALUATION COMPLETE")
        print("="*80)
    else:
        print("\n❌ No models were successfully evaluated")


if __name__ == "__main__":
    main()

