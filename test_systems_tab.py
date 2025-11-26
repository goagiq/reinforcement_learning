#!/usr/bin/env python3
"""
Comprehensive test suite for Systems Tab components.
Focuses especially on Adaptive Learning given the 0% win rate issue.
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

BASE_URL = "http://localhost:8200"
PROJECT_ROOT = Path(__file__).parent

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_test(name: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Testing: {name}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message: str):
    print(f"{Colors.GREEN}[PASS] {message}{Colors.END}")

def print_error(message: str):
    print(f"{Colors.RED}[FAIL] {message}{Colors.END}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}[WARN] {message}{Colors.END}")

def print_info(message: str):
    print(f"  {message}")

def test_api_endpoint(method: str, endpoint: str, expected_status: int = 200, data: Dict = None, params: Dict = None, description: str = None) -> Optional[Dict[Any, Any]]:
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, params=params, timeout=10)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == expected_status:
            print_success(f"{description or endpoint}: Status {response.status_code}")
            try:
                return response.json()
            except:
                return {"raw": response.text}
        else:
            print_error(f"{description or endpoint}: Expected {expected_status}, got {response.status_code}")
            print_info(f"Response: {response.text[:300]}")
            return None
    except requests.exceptions.ConnectionError:
        print_error(f"{description or endpoint}: Could not connect to server. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        print_error(f"{description or endpoint}: Request timeout")
        return None
    except Exception as e:
        print_error(f"{description or endpoint}: {str(e)}")
        return None

def test_component_1_adaptive_learning():
    """Test 1: Adaptive Learning Component (FOCUS)"""
    print_test("Component 1: Adaptive Learning (CRITICAL - 0% Win Rate Issue)")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        print_error("Could not get systems status")
        return
    
    components = result.get('components', {})
    adaptive_learning = components.get('adaptive_learning', {})
    
    print_info(f"Status: {adaptive_learning.get('status', 'unknown')}")
    print_info(f"Message: {adaptive_learning.get('message', 'N/A')}")
    
    # Check current parameters
    current_params = adaptive_learning.get('current_parameters', {})
    if current_params:
        print_success("Current parameters found")
        if 'entropy_coef' in current_params:
            print_info(f"  Entropy Coefficient: {current_params.get('entropy_coef')}")
        if 'inaction_penalty' in current_params:
            print_info(f"  Inaction Penalty: {current_params.get('inaction_penalty')}")
        if 'learning_rate' in current_params:
            print_info(f"  Learning Rate: {current_params.get('learning_rate')}")
        if 'min_risk_reward_ratio' in current_params:
            print_info(f"  Min Risk/Reward Ratio: {current_params.get('min_risk_reward_ratio')}")
    else:
        print_warning("No current parameters found")
    
    # Check last adjustment
    last_adjustment = adaptive_learning.get('last_adjustment')
    if last_adjustment:
        print_success("Last adjustment found")
        print_info(f"  Timestamp: {last_adjustment.get('timestamp', 'N/A')}")
        print_info(f"  Episode: {last_adjustment.get('episode', 'N/A')}")
        print_info(f"  Timestep: {last_adjustment.get('timestep', 'N/A')}")
        adjustments = last_adjustment.get('adjustments', {})
        if adjustments:
            print_info(f"  Adjustments made:")
            for key, value in adjustments.items():
                print_info(f"    - {key}: {value}")
        else:
            print_warning("  No adjustments details in last adjustment")
    else:
        print_warning("No last adjustment found - adaptive learning may not have made any adjustments yet")
    
    # Check total adjustments
    total_adjustments = adaptive_learning.get('total_adjustments', 0)
    print_info(f"Total adjustments: {total_adjustments}")
    
    # Critical check: Is adaptive learning active?
    if adaptive_learning.get('status') == 'active':
        print_success("Adaptive Learning is ACTIVE")
        
        # Check if it should be making adjustments given the poor performance
        print_info("\n[CRITICAL ANALYSIS] Given 0% win rate (266 losing trades), adaptive learning should:")
        print_info("  1. Detect poor performance")
        print_info("  2. Increase entropy_coef (encourage exploration)")
        print_info("  3. Increase inaction_penalty (penalize staying flat)")
        print_info("  4. Possibly adjust learning_rate")
        
        if total_adjustments == 0:
            print_error("[CRITICAL] Adaptive learning is active but has made NO adjustments!")
            print_error("  This is a problem - with 0% win rate, adjustments should have been triggered")
        else:
            print_success(f"Adaptive learning has made {total_adjustments} adjustment(s)")
            if not last_adjustment:
                print_warning("  But last adjustment details are missing - may be a display issue")
    else:
        print_error("[CRITICAL] Adaptive Learning is NOT ACTIVE")
        print_error("  This explains why no adjustments are being made despite poor performance!")
        print_info(f"  Status: {adaptive_learning.get('status')}")
        print_info(f"  Message: {adaptive_learning.get('message')}")

def test_component_2_adaptive_config_files():
    """Test 2: Adaptive Learning Configuration Files"""
    print_test("Component 2: Adaptive Learning Config Files")
    
    # Check if adaptive training config file exists
    adaptive_config_path = PROJECT_ROOT / "logs/adaptive_training/current_reward_config.json"
    config_adjustments_path = PROJECT_ROOT / "logs/adaptive_training/config_adjustments.jsonl"
    
    if adaptive_config_path.exists():
        print_success(f"Adaptive config file exists: {adaptive_config_path}")
        try:
            with open(adaptive_config_path, 'r') as f:
                config = json.load(f)
            print_info("Current adaptive configuration:")
            for key, value in config.items():
                print_info(f"  - {key}: {value}")
        except Exception as e:
            print_error(f"Failed to read adaptive config: {e}")
    else:
        print_warning(f"Adaptive config file not found: {adaptive_config_path}")
        print_info("This suggests adaptive training may not be enabled or initialized")
    
    if config_adjustments_path.exists():
        print_success(f"Adjustments history file exists: {config_adjustments_path}")
        try:
            with open(config_adjustments_path, 'r') as f:
                lines = f.readlines()
            print_info(f"Total adjustment entries: {len(lines)}")
            if len(lines) > 0:
                # Read last few adjustments
                print_info("Recent adjustments:")
                for line in lines[-5:]:  # Last 5 adjustments
                    if line.strip():
                        try:
                            adj = json.loads(line.strip())
                            print_info(f"  Episode {adj.get('episode', 'N/A')}: {adj.get('adjustments', {})}")
                        except:
                            pass
            else:
                print_warning("Adjustments file exists but is empty - no adjustments made yet")
        except Exception as e:
            print_error(f"Failed to read adjustments file: {e}")
    else:
        print_warning(f"Adjustments history file not found: {config_adjustments_path}")

def test_component_3_training_system():
    """Test 3: Training System Component"""
    print_test("Component 3: Training System")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        return
    
    components = result.get('components', {})
    training_system = components.get('training_system', {})
    
    print_info(f"Status: {training_system.get('status', 'unknown')}")
    print_info(f"Message: {training_system.get('message', 'N/A')}")
    
    gpu_status = training_system.get('gpu_status')
    if gpu_status:
        print_info(f"GPU Status: {gpu_status}")

def test_component_4_trading_system():
    """Test 4: Trading System Component"""
    print_test("Component 4: Trading System")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        return
    
    components = result.get('components', {})
    trading_system = components.get('trading_system', {})
    
    print_info(f"Status: {trading_system.get('status', 'unknown')}")
    print_info(f"Message: {trading_system.get('message', 'N/A')}")

def test_component_5_data_pipeline():
    """Test 5: Data Pipeline Component"""
    print_test("Component 5: Data Pipeline")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        return
    
    components = result.get('components', {})
    data_pipeline = components.get('data_pipeline', {})
    
    print_info(f"Status: {data_pipeline.get('status', 'unknown')}")
    print_info(f"Message: {data_pipeline.get('message', 'N/A')}")
    
    file_count = data_pipeline.get('file_count')
    if file_count is not None:
        print_info(f"Data files: {file_count}")

def test_component_6_environment():
    """Test 6: Environment Component"""
    print_test("Component 6: Environment")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        return
    
    components = result.get('components', {})
    environment = components.get('environment', {})
    
    print_info(f"Status: {environment.get('status', 'unknown')}")
    print_info(f"Message: {environment.get('message', 'N/A')}")
    
    config = environment.get('config', {})
    if config:
        print_info("Environment configuration:")
        for key, value in config.items():
            print_info(f"  - {key}: {value}")

def test_component_7_decision_gate():
    """Test 7: Decision Gate Component"""
    print_test("Component 7: Decision Gate")
    
    result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not result or result.get('status') != 'success':
        return
    
    components = result.get('components', {})
    decision_gate = components.get('decision_gate', {})
    
    print_info(f"Status: {decision_gate.get('status', 'unknown')}")
    print_info(f"Message: {decision_gate.get('message', 'N/A')}")
    
    quality_filters = decision_gate.get('quality_filters', {})
    if quality_filters:
        print_info("Quality filters:")
        print_info(f"  - Min Confidence: {quality_filters.get('min_action_confidence', 'N/A')}")
        print_info(f"  - Min Quality: {quality_filters.get('min_quality_score', 'N/A')}")

def analyze_adaptive_learning_status():
    """Deep analysis of Adaptive Learning status"""
    print_test("Deep Analysis: Adaptive Learning & Performance Correlation")
    
    # Get current trading metrics
    perf_result = test_api_endpoint("GET", "/api/monitoring/performance", description="Get current performance")
    win_rate = 0.0
    total_trades = 0
    if perf_result and perf_result.get('status') == 'success':
        metrics = perf_result.get('metrics', {})
        win_rate = metrics.get('win_rate', 0.0)
        total_trades = metrics.get('total_trades', 0)
        print_info(f"Current Performance: {total_trades} trades, {(win_rate * 100):.1f}% win rate")
    
    # Get systems status
    systems_result = test_api_endpoint("GET", "/api/systems/status", description="Get systems status")
    if not systems_result:
        return
    
    adaptive_learning = systems_result.get('components', {}).get('adaptive_learning', {})
    adaptive_status = adaptive_learning.get('status')
    total_adjustments = adaptive_learning.get('total_adjustments', 0)
    
    print_info("\n[ANALYSIS]")
    print_info(f"Win Rate: {(win_rate * 100):.1f}%")
    print_info(f"Total Trades: {total_trades}")
    print_info(f"Adaptive Learning Status: {adaptive_status}")
    print_info(f"Total Adjustments: {total_adjustments}")
    
    # Critical assessment
    if win_rate == 0.0 and total_trades > 50:
        print_error("\n[CRITICAL ISSUE DETECTED]")
        print_error(f"  - 0% win rate with {total_trades} trades is extremely poor")
        print_error(f"  - Adaptive learning should be making aggressive adjustments")
        
        if adaptive_status != 'active':
            print_error(f"  - PROBLEM: Adaptive learning is NOT ACTIVE ({adaptive_status})")
            print_info("  - SOLUTION: Check if adaptive training is enabled in config")
        elif total_adjustments == 0:
            print_error(f"  - PROBLEM: Adaptive learning is active but made NO adjustments")
            print_info("  - SOLUTION: Check adaptive training evaluation frequency and thresholds")
        else:
            print_warning(f"  - Adaptive learning has made {total_adjustments} adjustments")
            print_warning(f"  - But win rate is still 0% - adjustments may not be effective")
            print_info("  - Check if adjustments are actually being applied to the agent")
    
    # Check if adaptive training config exists and is being used
    config_path = PROJECT_ROOT / "logs/adaptive_training/current_reward_config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print_info(f"\nCurrent Adaptive Config Values:")
            for key, value in config.items():
                print_info(f"  {key}: {value}")
        except Exception as e:
            print_error(f"Failed to read config: {e}")

def run_all_tests():
    """Run all component tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SYSTEMS TAB COMPONENT TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}Focus: Adaptive Learning (0% Win Rate Issue){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"\nTesting backend API endpoints for Systems Tab components...")
    print(f"Base URL: {BASE_URL}\n")
    
    # Test all components
    test_component_1_adaptive_learning()
    test_component_2_adaptive_config_files()
    analyze_adaptive_learning_status()  # Deep analysis
    test_component_3_training_system()
    test_component_4_trading_system()
    test_component_5_data_pipeline()
    test_component_6_environment()
    test_component_7_decision_gate()
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"\n[PASS] Core API endpoints tested")
    print(f"[INFO] Adaptive Learning component analyzed in detail")
    print(f"\n{Colors.BOLD}Key Findings:{Colors.END}")
    print(f"1. Check if Adaptive Learning status is 'active'")
    print(f"2. Check if adjustments are being made (total_adjustments > 0)")
    print(f"3. Review adaptive training config files")
    print(f"4. Verify adaptive trainer is initialized in training config")
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

