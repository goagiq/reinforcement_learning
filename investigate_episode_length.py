"""
Investigate why episodes are terminating early (60 steps vs 10,000 expected)
This script analyzes the episode termination logic without impacting training.
"""

import yaml
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def investigate_episode_termination():
    """Investigate episode termination logic"""
    print("\n" + "=" * 60)
    print("EPISODE TERMINATION INVESTIGATION")
    print("=" * 60)
    
    # Load config
    config_path = project_root / "configs" / "train_config_adaptive.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n[1] CONFIGURATION CHECK")
    print("-" * 60)
    max_episode_steps = config["environment"].get("max_episode_steps", 10000)
    lookback_bars = config["environment"].get("lookback_bars", 20)
    
    print(f"  max_episode_steps: {max_episode_steps}")
    print(f"  lookback_bars: {lookback_bars}")
    
    if max_episode_steps != 10000:
        print(f"  [WARN] max_episode_steps is not 10,000!")
    
    print("\n[2] TRADING ENVIRONMENT TERMINATION LOGIC")
    print("-" * 60)
    
    # Read trading_env.py to check termination logic
    trading_env_path = project_root / "src" / "trading_env.py"
    with open(trading_env_path, 'r') as f:
        trading_env_code = f.read()
    
    # Check for termination conditions
    termination_checks = []
    
    # Check for max_steps termination
    if "current_step >= self.max_steps" in trading_env_code:
        termination_checks.append(("max_steps", "Episode terminates when current_step >= max_steps"))
    
    if "terminated = self.current_step >= self.max_steps" in trading_env_code:
        termination_checks.append(("max_steps_explicit", "Explicit check: current_step >= max_steps"))
    
    # Check for truncated
    if "truncated" in trading_env_code:
        truncated_lines = [i+1 for i, line in enumerate(trading_env_code.split('\n')) if 'truncated' in line.lower() and '=' in line]
        if truncated_lines:
            termination_checks.append(("truncated", f"Truncated flag set (lines: {truncated_lines[:3]})"))
    
    # Check for early termination conditions
    early_termination_keywords = [
        "done = True",
        "terminated = True",
        "truncated = True",
        "early_stop",
        "early termination",
        "break",
        "return"
    ]
    
    for keyword in early_termination_keywords:
        if keyword in trading_env_code:
            lines = [i+1 for i, line in enumerate(trading_env_code.split('\n')) if keyword.lower() in line.lower()]
            if lines:
                termination_checks.append((keyword, f"Found '{keyword}' at lines: {lines[:5]}"))
    
    print("  Termination conditions found:")
    for check_type, description in termination_checks:
        print(f"    - {check_type}: {description}")
    
    print("\n[3] POSSIBLE CAUSES OF SHORT EPISODES")
    print("-" * 60)
    
    causes = []
    
    # Check if data might be running out
    print("  1. Data Length Issue:")
    print("     - If data is shorter than max_episode_steps + lookback_bars,")
    print("       episodes will terminate early")
    print(f"     - Required data length: {max_episode_steps + lookback_bars} bars")
    print(f"     - Check if data files have enough bars")
    
    causes.append(("data_length", "Data might be shorter than required"))
    
    # Check for errors/exceptions
    print("\n  2. Error/Exception Handling:")
    print("     - If an exception occurs during step(), episode might terminate")
    print("     - Check for try/except blocks that might catch errors")
    print("     - Check if errors are being silently handled")
    
    causes.append(("exceptions", "Errors might be causing early termination"))
    
    # Check for boundary conditions
    print("\n  3. Boundary Conditions:")
    print("     - If current_step reaches data length before max_steps")
    print("     - If lookback_bars causes issues at start of episode")
    print(f"     - Check: current_step + lookback_bars < data_length")
    
    causes.append(("boundary", "Boundary conditions might cause early termination"))
    
    # Check for reset issues
    print("\n  4. Reset Logic:")
    print("     - If reset() is called unexpectedly")
    print("     - If episode tracking is reset incorrectly")
    print("     - Check reset() implementation")
    
    causes.append(("reset", "Reset logic might be causing issues"))
    
    print("\n[4] DIAGNOSTIC SUGGESTIONS")
    print("-" * 60)
    
    print("  To diagnose without impacting training:")
    print("\n  1. Check data file lengths:")
    print("     - Verify data files have enough bars")
    print("     - Required: max_episode_steps + lookback_bars = {} bars".format(max_episode_steps + lookback_bars))
    
    print("\n  2. Add diagnostic logging (non-intrusive):")
    print("     - Add logging to step() method to track:")
    print("       * current_step value")
    print("       * terminated/truncated flags")
    print("       * data length checks")
    print("     - This can be done in a separate branch/file")
    
    print("\n  3. Check for exceptions in logs:")
    print("     - Look for error messages in training logs")
    print("     - Check if exceptions are being caught silently")
    
    print("\n  4. Verify episode reset logic:")
    print("     - Check if reset() is being called correctly")
    print("     - Verify episode tracking variables")
    
    print("\n[5] CODE INSPECTION")
    print("-" * 60)
    
    # Read specific sections of trading_env.py
    print("\n  Checking step() method termination logic...")
    
    # Find the step method
    step_method_start = trading_env_code.find("def step(")
    if step_method_start != -1:
        # Find the end of the step method (next def or class)
        step_method_end = trading_env_code.find("\n    def ", step_method_start + 1)
        if step_method_end == -1:
            step_method_end = trading_env_code.find("\nclass ", step_method_start + 1)
        if step_method_end == -1:
            step_method_end = len(trading_env_code)
        
        step_method = trading_env_code[step_method_start:step_method_end]
        
        # Check for termination conditions in step method
        if "terminated = self.current_step >= self.max_steps" in step_method:
            print("    [FOUND] Termination check: current_step >= max_steps")
            # Find the line number
            lines_before = trading_env_code[:step_method_start].count('\n')
            line_in_method = step_method[:step_method.find("terminated = self.current_step >= self.max_steps")].count('\n')
            line_number = lines_before + line_in_method + 1
            print(f"    Line number: ~{line_number}")
        
        if "truncated = False" in step_method:
            print("    [FOUND] truncated is set to False (no early truncation)")
        
        if "truncated = True" in step_method:
            print("    [WARN] truncated is set to True somewhere - this could cause early termination")
    
    print("\n  Checking reset() method...")
    reset_method_start = trading_env_code.find("def reset(")
    if reset_method_start != -1:
        reset_method_end = trading_env_code.find("\n    def ", reset_method_start + 1)
        if reset_method_end == -1:
            reset_method_end = trading_env_code.find("\nclass ", reset_method_start + 1)
        if reset_method_end == -1:
            reset_method_end = len(trading_env_code)
        
        reset_method = trading_env_code[reset_method_start:reset_method_end]
        
        # Check for issues in reset
        if "current_step" in reset_method:
            print("    [FOUND] reset() modifies current_step")
            if "self.current_step = self.lookback_bars" in reset_method:
                print("    [OK] current_step is set to lookback_bars (correct)")
            else:
                print("    [WARN] current_step might not be set correctly")
    
    print("\n[6] RECOMMENDATIONS")
    print("-" * 60)
    
    print("\n  1. [SAFE] Check data file lengths:")
    print("     - Run: python -c \"import pandas as pd; df = pd.read_csv('data/raw/ES_1min.csv'); print(f'Data length: {len(df)} bars')\"")
    print(f"     - Required: {max_episode_steps + lookback_bars} bars minimum")
    
    print("\n  2. [SAFE] Review training logs:")
    print("     - Look for '[DEBUG]' messages about episode completion")
    print("     - Check for error messages or exceptions")
    print("     - Look for patterns in episode lengths")
    
    print("\n  3. [SAFE] Create diagnostic script:")
    print("     - Create a script that loads the environment")
    print("     - Run a few episodes and track step counts")
    print("     - This won't impact running training")
    
    print("\n  4. [NON-INTRUSIVE] Add diagnostic logging:")
    print("     - Can add logging to a separate file")
    print("     - Or add conditional logging (only if debug flag is set)")
    print("     - This won't impact training performance")
    
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check data file lengths")
    print("  2. Review training logs for patterns")
    print("  3. Create diagnostic script to test environment")
    print("  4. Consider adding non-intrusive logging")

if __name__ == "__main__":
    investigate_episode_termination()

