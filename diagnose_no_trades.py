"""
Diagnostic script to identify why no trades are occurring.
Checks all thresholds and filters that could block trades.
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

def diagnose_no_trades():
    """Diagnose why no trades are occurring"""
    print("\n" + "=" * 60)
    print("DIAGNOSIS: Why No Trades Are Occurring")
    print("=" * 60)
    
    # Load config
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print("[ERROR] Config file not found")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n[1] ACTION THRESHOLD (TradingEnvironment)")
    print("-" * 60)
    action_threshold = config.get("environment", {}).get("action_threshold", 0.05)
    print(f"  action_threshold: {action_threshold}")
    print(f"  Impact: Only actions with |position_change| > {action_threshold} trigger trades")
    if action_threshold >= 0.05:
        print(f"  [WARN] Threshold is high ({action_threshold}) - may filter many trades")
    
    print("\n[2] DECISIONGATE FILTERS")
    print("-" * 60)
    decision_gate = config.get("decision_gate", {})
    use_decision_gate = config.get("training", {}).get("use_decision_gate", False)
    print(f"  use_decision_gate: {use_decision_gate}")
    
    if use_decision_gate:
        min_combined_confidence = decision_gate.get("min_combined_confidence", 0.7)
        min_confluence_required = decision_gate.get("min_confluence_required", 2)
        swarm_enabled = decision_gate.get("swarm_enabled", True)
        
        print(f"  min_combined_confidence: {min_combined_confidence}")
        print(f"  min_confluence_required: {min_confluence_required}")
        print(f"  swarm_enabled: {swarm_enabled}")
        
        if swarm_enabled and min_confluence_required >= 2:
            print(f"  [WARN] Swarm enabled with min_confluence_required={min_confluence_required}")
            print(f"         During training (RL-only), confluence_count=0, so ALL trades will be rejected!")
        
        if min_combined_confidence >= 0.7:
            print(f"  [WARN] min_combined_confidence={min_combined_confidence} is high")
            print(f"         Actions with confidence < {min_combined_confidence} will be rejected")
        
        quality_scorer = decision_gate.get("quality_scorer", {})
        if quality_scorer.get("enabled", False):
            min_quality_score = quality_scorer.get("min_quality_score", 0.6)
            print(f"  quality_scorer.enabled: True")
            print(f"  min_quality_score: {min_quality_score}")
            if min_quality_score >= 0.6:
                print(f"  [WARN] min_quality_score={min_quality_score} is high")
    
    print("\n[3] QUALITY FILTERS (TradingEnvironment)")
    print("-" * 60)
    quality_filters = config.get("environment", {}).get("reward", {}).get("quality_filters", {})
    enabled = quality_filters.get("enabled", False)
    print(f"  quality_filters.enabled: {enabled}")
    
    if enabled:
        min_action_confidence = quality_filters.get("min_action_confidence", 0.3)
        min_quality_score = quality_filters.get("min_quality_score", 0.5)
        require_positive_ev = quality_filters.get("require_positive_expected_value", True)
        
        print(f"  min_action_confidence: {min_action_confidence}")
        print(f"  min_quality_score: {min_quality_score}")
        print(f"  require_positive_expected_value: {require_positive_ev}")
        
        if min_action_confidence >= 0.3:
            print(f"  [WARN] min_action_confidence={min_action_confidence} may filter many actions")
        if min_quality_score >= 0.5:
            print(f"  [WARN] min_quality_score={min_quality_score} may filter many trades")
        if require_positive_ev:
            print(f"  [WARN] require_positive_expected_value=True - trades with EV <= 0 will be rejected")
            print(f"         Early in training, EV calculation may not be accurate")
    
    print("\n[4] CONSECUTIVE LOSS LIMIT")
    print("-" * 60)
    max_consecutive_losses = config.get("environment", {}).get("reward", {}).get("max_consecutive_losses", 3)
    print(f"  max_consecutive_losses: {max_consecutive_losses}")
    print(f"  Impact: Trading pauses after {max_consecutive_losses} consecutive losses")
    
    print("\n[5] RECOMMENDATIONS")
    print("-" * 60)
    print("  To allow trades during training, consider:")
    print()
    
    issues_found = []
    
    if use_decision_gate and swarm_enabled and min_confluence_required >= 2:
        issues_found.append("CRITICAL")
        print("  [CRITICAL] DecisionGate with swarm_enabled=True and min_confluence_required>=2")
        print("             will reject ALL RL-only trades (confluence_count=0)")
        print("             SOLUTION: Set swarm_enabled: false OR min_confluence_required: 0")
    
    if use_decision_gate and min_combined_confidence >= 0.7:
        issues_found.append("HIGH")
        print("  [HIGH] min_combined_confidence is high - may reject many trades")
        print("         SOLUTION: Reduce to 0.5-0.6 for training")
    
    if action_threshold >= 0.05:
        issues_found.append("MEDIUM")
        print("  [MEDIUM] action_threshold is high - may filter many small actions")
        print("           SOLUTION: Reduce to 0.02-0.03 for training")
    
    if enabled and min_action_confidence >= 0.3:
        issues_found.append("MEDIUM")
        print("  [MEDIUM] quality_filters.min_action_confidence is high")
        print("           SOLUTION: Reduce to 0.1-0.2 for training")
    
    if enabled and require_positive_ev:
        issues_found.append("MEDIUM")
        print("  [MEDIUM] require_positive_expected_value=True may reject trades early in training")
        print("           SOLUTION: Set to false OR ensure EV calculation has enough data")
    
    if not issues_found:
        print("  [OK] No obvious issues found - may be agent not generating actions")
        print("       Check if agent is exploring (entropy_coef) and generating non-zero actions")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_no_trades()

