"""
Quick script to check if training is progressing properly.
Shows timestep and episode progression over time.
"""
import requests
import time
import sys

def check_progress():
    """Check training progress and verify it's advancing"""
    try:
        response = requests.get('http://localhost:8200/api/training/status', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "running":
            print(f"[WARN] Training status: {data.get('status')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            return False
        
        metrics = data.get("metrics", {})
        timestep = metrics.get("timestep", 0)
        episode = metrics.get("episode", 0)
        completed_episodes = metrics.get("completed_episodes", 0)
        current_length = metrics.get("current_episode_length", 0)
        progress = metrics.get("progress_percent", 0)
        
        print(f"[OK] Training is RUNNING")
        print(f"   Timestep: {timestep:,} / {metrics.get('total_timesteps', 0):,} ({progress:.2f}%)")
        print(f"   Episode: {episode} (completed: {completed_episodes})")
        print(f"   Current episode length: {current_length:,} steps")
        print(f"   Latest reward: {metrics.get('latest_reward', 0):.2f}")
        
        # Check if episode is progressing
        if current_length > 0:
            print(f"   [OK] Episode is active and progressing")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error connecting to API: {e}")
        print(f"   Make sure the backend server is running on http://localhost:8200")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    print("Checking training progress...\n")
    
    # Check once
    if check_progress():
        print("\n[INFO] Tip: Run this script multiple times to verify timestep/episode are increasing")
        print("   Example: python scripts/check_training_progress.py")
    
    sys.exit(0 if check_progress() else 1)

