"""
Check for error messages in training console output or logs
"""

import sys
from pathlib import Path
import re

def check_for_errors():
    """Check for error messages in various locations"""
    print("=" * 80)
    print("CHECKING FOR TRAINING ERRORS")
    print("=" * 80)
    print()
    
    errors_found = []
    
    # Error patterns to search for
    error_patterns = [
        r'\[ERROR\].*Exception.*env\.step',
        r'\[ERROR\].*Exception.*_get_state_features',
        r'\[WARNING\].*Episode terminating early',
        r'UnboundLocalError',
        r'IndexError',
        r'KeyError',
        r'Exception in env\.step',
        r'Exception in _get_state_features',
    ]
    
    # 1. Check training summary
    print("1. Checking training summary...")
    summary_path = Path("logs/training_summary.json")
    if summary_path.exists():
        try:
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"   [OK] Training summary found")
            print(f"   Total episodes: {summary.get('total_episodes', 'N/A')}")
            print(f"   Mean episode length: {summary.get('mean_episode_length', 'N/A')}")
            if summary.get('mean_episode_length', 0) < 1000:
                print(f"   [WARNING] Mean episode length is very short!")
                errors_found.append("Mean episode length is very short")
        except Exception as e:
            print(f"   [ERROR] Could not read summary: {e}")
    else:
        print("   [INFO] No training summary found")
    
    # 2. Check for recent log files
    print("\n2. Checking for log files...")
    logs_dir = Path("logs")
    if logs_dir.exists():
        # Find most recent training session
        training_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_training_")]
        if training_dirs:
            latest_dir = max(training_dirs, key=lambda d: d.stat().st_mtime)
            print(f"   [OK] Found training directory: {latest_dir.name}")
            
            # Check for any text files or event files
            log_files = list(latest_dir.glob("*.log")) + list(latest_dir.glob("*.txt"))
            if log_files:
                print(f"   [OK] Found {len(log_files)} log file(s)")
                for log_file in log_files[:3]:  # Check first 3
                    print(f"   Checking: {log_file.name}")
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for pattern in error_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    print(f"      [FOUND] Pattern '{pattern}': {len(matches)} matches")
                                    errors_found.extend(matches[:3])  # First 3 matches
                    except Exception as e:
                        print(f"      [ERROR] Could not read: {e}")
            else:
                print("   [INFO] No .log or .txt files found")
        else:
            print("   [INFO] No training directories found")
    else:
        print("   [INFO] Logs directory does not exist")
    
    # 3. Instructions for checking console
    print("\n3. Console Output Check:")
    print("   [INFO] To check console output during training:")
    print("   - If training is running, check the console/terminal window")
    print("   - Look for messages starting with [ERROR] or [WARNING]")
    print("   - Common error messages:")
    print("     * [ERROR] Exception in env.step()")
    print("     * [ERROR] Exception in _get_state_features")
    print("     * [WARNING] Episode terminating early")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if errors_found:
        print(f"[FOUND] {len(errors_found)} potential error(s) detected:")
        for i, error in enumerate(errors_found[:10], 1):  # First 10
            print(f"   {i}. {error}")
    else:
        print("[OK] No obvious errors found in logs")
        print("   Note: This doesn't mean there are no errors")
        print("   Check console output during training for [ERROR] messages")
    
    print("\nNext steps:")
    print("1. Check console output if training is currently running")
    print("2. Look for [ERROR] messages in the terminal")
    print("3. Check if episodes are terminating at step 20")
    print("4. Review the test_episode_termination.py results")

if __name__ == "__main__":
    check_for_errors()

