"""
Check Adaptive Learning Status - Fixed Version
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.colors import success, info, warn, error

def check_adaptive_learning():
    """Comprehensive adaptive learning status check"""
    
    print(info("\n" + "="*70))
    print(info("ADAPTIVE LEARNING STATUS CHECK"))
    print(info("="*70))
    
    # Check config
    config_path = project_root / "logs/adaptive_training/current_reward_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(success("\n[CONFIG] Current Adaptive Parameters:"))
        print(info(f"  R:R Ratio: {config.get('min_risk_reward_ratio', 'N/A')}"))
        print(info(f"  Entropy Coef: {config.get('entropy_coef', 'N/A')}"))
        print(info(f"  Inaction Penalty: {config.get('inaction_penalty', 'N/A')}"))
        if 'quality_filters' in config:
            qf = config['quality_filters']
            print(info(f"  Min Action Confidence: {qf.get('min_action_confidence', 'N/A')}"))
            print(info(f"  Min Quality Score: {qf.get('min_quality_score', 'N/A')}"))
        if 'stop_loss_pct' in config:
            print(info(f"  Stop Loss: {config['stop_loss_pct']*100:.1f}%"))
    else:
        print(warn("[WARN] Adaptive config file not found"))
    
    # Check adjustments
    adjustments_file = project_root / "logs/adaptive_training/config_adjustments.jsonl"
    if adjustments_file.exists():
        lines = adjustments_file.read_text().strip().split('\n')
        all_adjustments = [json.loads(line) for line in lines if line.strip()]
        
        print(success(f"\n[ADJUSTMENTS] Total: {len(all_adjustments)}"))
        
        if all_adjustments:
            last_adj = all_adjustments[-1]
            print(info(f"\n[LAST ADJUSTMENT]"))
            print(info(f"  Timestep: {last_adj.get('timestep', 'N/A')}"))
            print(info(f"  Episode: {last_adj.get('episode', 'N/A')}"))
            print(info(f"  Time: {last_adj.get('timestamp', 'N/A')}"))
            
            adj_details = last_adj.get('adjustments', {})
            if adj_details:
                print(info(f"  Changes:"))
                for key, value in adj_details.items():
                    if isinstance(value, dict):
                        print(info(f"    {key}:"))
                        for k, v in value.items():
                            print(info(f"      {k}: {v}"))
                    else:
                        print(info(f"    {key}: {value}"))
            
            # Recent adjustments (last 10)
            recent = all_adjustments[-10:]
            print(success(f"\n[RECENT ADJUSTMENTS] (Last 10)"))
            for i, adj in enumerate(recent, 1):
                ts = adj.get('timestep', 0)
                ep = adj.get('episode', 0)
                adj_count = len(adj.get('adjustments', {}))
                print(info(f"  {i}. Timestep {ts}, Episode {ep}, {adj_count} adjustment(s)"))
            
            # Adjustment types
            entropy_count = sum(1 for a in all_adjustments if 'entropy_coef' in a.get('adjustments', {}))
            rr_count = sum(1 for a in all_adjustments if 'min_risk_reward_ratio' in a.get('adjustments', {}))
            quality_count = sum(1 for a in all_adjustments if 'quality_filters' in a.get('adjustments', {}))
            inaction_count = sum(1 for a in all_adjustments if 'inaction_penalty' in a.get('adjustments', {}))
            
            print(success(f"\n[ADJUSTMENT TYPES]"))
            print(info(f"  Entropy Coef: {entropy_count}"))
            print(info(f"  R:R Ratio: {rr_count}"))
            print(info(f"  Quality Filters: {quality_count}"))
            print(info(f"  Inaction Penalty: {inaction_count}"))
    else:
        print(warn("[WARN] Adjustments file not found"))
    
    # Check training status
    model_dir = project_root / "models"
    checkpoints = sorted([cp for cp in model_dir.glob("checkpoint_*.pt")], 
                        key=lambda x: int(x.stem.split('_')[1]), reverse=True)
    
    if checkpoints:
        latest = checkpoints[0]
        latest_ts = int(latest.stem.split('_')[1])
        print(success(f"\n[TRAINING STATUS]"))
        print(info(f"  Latest checkpoint: {latest.name}"))
        print(info(f"  Latest timestep: {latest_ts:,}"))
        
        if all_adjustments:
            last_ts = all_adjustments[-1].get('timestep', 0)
            ts_since = latest_ts - last_ts
            print(info(f"  Timesteps since last adjustment: {ts_since:,}"))
            
            if ts_since > 5000:
                print(warn(f"  [WARN] No adjustment for {ts_since:,} timesteps (eval frequency: 5,000)"))
            else:
                print(success(f"  [OK] Recent adjustment activity"))
    
    print(info("\n" + "="*70))


if __name__ == "__main__":
    check_adaptive_learning()

