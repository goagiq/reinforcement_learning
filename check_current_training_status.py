"""
Check current training status and verify if Priority 1 features are active
"""

import requests
import json
from datetime import datetime

def check_training_status():
    """Check training status via API"""
    
    try:
        # Try to get training status
        response = requests.get("http://localhost:8200/api/training/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("=" * 60)
            print("CURRENT TRAINING STATUS")
            print("=" * 60)
            print(f"Status: {status.get('status', 'unknown')}")
            print(f"Message: {status.get('message', 'N/A')}")
            
            metrics = status.get('metrics', {})
            if metrics:
                print(f"\nTraining Metrics:")
                print(f"  Timestep: {metrics.get('timestep', 'N/A')}")
                print(f"  Episode: {metrics.get('episode', 'N/A')}")
                print(f"  Mean Reward: {metrics.get('mean_reward', 'N/A')}")
            
            # Check if there's an error
            if status.get('error'):
                print(f"\n[ERROR] {status.get('error')}")
            
            # Check if there's a warning
            if status.get('warning'):
                print(f"\n[WARN] {status.get('warning')}")
            
            return status
        else:
            print(f"[ERROR] API returned status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API server at http://localhost:8200")
        print("  - Is the API server running?")
        print("  - Check if training was started via command line instead")
        return None
    except Exception as e:
        print(f"[ERROR] Error checking training status: {e}")
        return None

def check_config_used():
    """Try to determine which config was used"""
    print("\n" + "=" * 60)
    print("CONFIG VERIFICATION")
    print("=" * 60)
    
    # Check if we can get config info from API
    try:
        response = requests.get("http://localhost:8200/api/configs", timeout=5)
        if response.status_code == 200:
            configs = response.json()
            print(f"Available configs: {len(configs.get('configs', []))}")
            
            # Check for train_config_adaptive.yaml
            config_files = [c.get('name', '') for c in configs.get('configs', [])]
            if 'train_config_adaptive.yaml' in config_files:
                print("[OK] train_config_adaptive.yaml is available")
            else:
                print("[WARN] train_config_adaptive.yaml not found in available configs")
    except:
        print("[INFO] Could not check configs via API")

if __name__ == "__main__":
    print(f"\nTraining Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    status = check_training_status()
    check_config_used()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if status and status.get('status') == 'running':
        print("[INFO] Training is currently running")
        print("[INFO] To verify Priority 1 features are active:")
        print("  1. Check the console/terminal where training was started")
        print("  2. Look for these messages:")
        print("     'Creating trading environment...'")
        print("     '  [OK] Slippage model: Enabled'")
        print("     '  [OK] Market impact model: Enabled'")
        print("     '  [OK] Execution quality tracker: Available'")
        print("\n  3. If you DON'T see these messages:")
        print("     - Training started before config update")
        print("     - Restart training to activate Priority 1 features")
    elif status and status.get('status') == 'starting':
        print("[INFO] Training is initializing...")
        print("[INFO] Wait for initialization to complete, then check console output")
    else:
        print("[INFO] No active training detected")
        print("[INFO] When you start training, check console output for Priority 1 feature status")

