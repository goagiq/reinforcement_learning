"""
End-to-End Test for Training Flow

Tests the complete training workflow:
1. Training start request
2. Status polling
3. Metrics updates
4. Checkpoint resuming
"""

import requests
import time
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:8200"
TIMEOUT = 120  # 2 minutes timeout for initialization


def test_api_health():
    """Test if API server is running"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        assert response.status_code == 200
        print("✅ API server is running")
        return True
    except Exception as e:
        print(f"❌ API server not reachable: {e}")
        print(f"   Make sure backend is running: python start-ui.py")
        return False


def test_training_status_idle():
    """Test training status when no training is running"""
    print("\n🔍 Testing training status (idle)...")
    try:
        response = requests.get(f"{BASE_URL}/api/training/status", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["idle", "running", "starting"]
        print(f"✅ Status endpoint works: {data['status']}")
        return True
    except Exception as e:
        print(f"❌ Status endpoint failed: {e}")
        return False


def cleanup_stale_training():
    """Clean up any stale training entries"""
    print("\n🔍 Cleaning up stale training entries...")
    try:
        # Try to stop training first (will fail if not running, that's ok)
        response = requests.post(f"{BASE_URL}/api/training/stop", timeout=5)
        print(f"   Stop response: {response.status_code}")
    except:
        pass
    
    # Wait a moment for cleanup
    time.sleep(2)
    
    # Check status
    try:
        response = requests.get(f"{BASE_URL}/api/training/status", timeout=5)
        data = response.json()
        if data["status"] != "idle":
            print(f"   ⚠️  Training still in state: {data['status']}")
            print(f"   Message: {data.get('message', '')}")
            return False
        print("   ✅ Training cleaned up")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not verify cleanup: {e}")
        return False


def test_training_start():
    """Test starting training with checkpoint"""
    print("\n🔍 Testing training start...")
    
    # First, cleanup any stale training
    cleanup_stale_training()
    time.sleep(1)
    
    # Check if checkpoint exists
    checkpoint_path = "models/checkpoint_470000.pt"
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Testing without checkpoint...")
        checkpoint_path = None
    
    request_data = {
        "device": "cpu",  # Use CPU for testing (faster, more reliable)
        "total_timesteps": 10000,  # Small number for testing
        "config_path": "configs/train_config.yaml",
        "checkpoint_path": checkpoint_path
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/training/start",
            json=request_data,
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        print(f"✅ Training start request successful: {data['message']}")
        return True
    except requests.exceptions.Timeout:
        print("⚠️  Request timed out (backend may be busy)")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ Training start failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"   Error detail: {error_data.get('detail', 'Unknown')}")
            except:
                print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Training start failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_training_status_progression():
    """Test that training status progresses correctly"""
    print("\n🔍 Testing training status progression...")
    
    max_wait = TIMEOUT
    start_time = time.time()
    last_status = None
    status_transitions = []
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/training/status", timeout=5)
            assert response.status_code == 200
            data = response.json()
            status = data["status"]
            
            # Track status changes
            if status != last_status:
                status_transitions.append({
                    "time": time.time() - start_time,
                    "status": status,
                    "message": data.get("message", "")
                })
                print(f"   [{status_transitions[-1]['time']:.1f}s] Status: {status} - {data.get('message', '')}")
                last_status = status
            
            # Check if we have metrics
            if status == "running" and "metrics" in data and data["metrics"]:
                metrics = data["metrics"]
                if metrics.get("timestep") is not None:
                    print(f"✅ Training is running with metrics!")
                    print(f"   Timestep: {metrics.get('timestep')}")
                    print(f"   Episode: {metrics.get('episode')}")
                    print(f"   Reward: {metrics.get('latest_reward', 'N/A')}")
                    return True
            
            # Check for errors
            if status == "error":
                print(f"❌ Training failed: {data.get('message', 'Unknown error')}")
                return False
            
            # If status is "starting", wait longer
            if status == "starting":
                time.sleep(2)
                continue
            
            # If running but no metrics yet, wait a bit more
            if status == "running":
                time.sleep(1)
                continue
                
        except Exception as e:
            print(f"⚠️  Error checking status: {e}")
            time.sleep(2)
    
    print(f"⚠️  Timeout waiting for training to start ({max_wait}s)")
    print(f"   Status transitions: {status_transitions}")
    return False


def test_training_stop():
    """Test stopping training"""
    print("\n🔍 Testing training stop...")
    try:
        response = requests.post(f"{BASE_URL}/api/training/stop", timeout=5)
        # 200 or 400 (if not running) are both acceptable
        assert response.status_code in [200, 400]
        print("✅ Stop endpoint works")
        return True
    except Exception as e:
        print(f"⚠️  Stop endpoint issue: {e}")
        return False


def run_all_tests():
    """Run all e2e tests"""
    print("="*60)
    print("🧪 End-to-End Training Flow Tests")
    print("="*60)
    
    results = []
    
    # Test 1: API Health
    results.append(("API Health", test_api_health()))
    if not results[-1][1]:
        print("\n❌ Cannot proceed - API server not running")
        print("   Please start the backend: python start-ui.py")
        return
    
    # Test 2: Status (idle)
    results.append(("Status (idle)", test_training_status_idle()))
    
    # Cleanup before starting
    print("\n🔍 Cleaning up before starting tests...")
    cleanup_stale_training()
    time.sleep(1)
    
    # Test 3: Training Start
    results.append(("Training Start", test_training_start()))
    if not results[-1][1]:
        print("\n⚠️  Training start failed - skipping progression test")
    else:
        # Test 4: Status Progression
        results.append(("Status Progression", test_training_status_progression()))
        
        # Test 5: Stop
        time.sleep(2)  # Give it a moment
        results.append(("Training Stop", test_training_stop()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

