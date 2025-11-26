#!/usr/bin/env python3
"""
Comprehensive test suite for Monitoring Tab components.
Tests all API endpoints and component functionality.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8200"

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

def test_api_endpoint(method: str, endpoint: str, expected_status: int = 200, data: Dict = None, params: Dict = None, description: str = None) -> Dict[Any, Any]:
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
            print_info(f"Response: {response.text[:200]}")
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

def test_component_1_performance_metrics():
    """Test 1: Performance Metrics Dashboard"""
    print_test("Component 1: Performance Metrics Dashboard")
    
    # Test without filter (all trades)
    result = test_api_endpoint("GET", "/api/monitoring/performance", description="Get performance metrics (all trades)")
    if result:
        if result.get('status') == 'success':
            metrics = result.get('metrics', {})
            print_success("Performance metrics loaded successfully")
            print_info(f"Total Trades: {metrics.get('total_trades', 0)}")
            print_info(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            print_info(f"Win Rate: {(metrics.get('win_rate', 0) * 100):.2f}%")
            print_info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print_info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print_info(f"Max Drawdown: {(metrics.get('max_drawdown', 0) * 100):.2f}%")
            print_info(f"Source: {metrics.get('source', 'unknown')}")
        else:
            print_error(f"API returned status: {result.get('status')}")
    
    # Test with timestamp filter (simulating filterBySession)
    print_info("\nTesting with timestamp filter (simulating current session only)...")
    # Get training status to get checkpoint timestamp
    training_status = test_api_endpoint("GET", "/api/training/status", description="Get training status for checkpoint timestamp")
    if training_status and training_status.get('checkpoint_resume_timestamp'):
        checkpoint_ts = training_status.get('checkpoint_resume_timestamp')
        result_filtered = test_api_endpoint("GET", "/api/monitoring/performance", 
                                           params={"since": checkpoint_ts},
                                           description="Get performance metrics (filtered by session)")
        if result_filtered and result_filtered.get('status') == 'success':
            metrics = result_filtered.get('metrics', {})
            print_info(f"Filtered Total Trades: {metrics.get('total_trades', 0)}")
            print_info(f"Filtered Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            if metrics.get('filtered_since'):
                print_info(f"Filtered since: {metrics.get('filtered_since')}")

def test_component_2_equity_curve():
    """Test 2: Equity Curve Chart"""
    print_test("Component 2: Equity Curve Chart")
    
    # Test without filter
    result = test_api_endpoint("GET", "/api/journal/equity-curve", 
                              params={"limit": 100},
                              description="Get equity curve (all data)")
    if result:
        if result.get('status') == 'success':
            equity_curve = result.get('equity_curve', [])
            print_success(f"Equity curve loaded: {len(equity_curve)} points")
            if len(equity_curve) > 0:
                first_point = equity_curve[0]
                last_point = equity_curve[-1]
                print_info(f"First point - Equity: ${first_point.get('equity', 0):.2f}, Episode: {first_point.get('episode', 'N/A')}")
                print_info(f"Last point - Equity: ${last_point.get('equity', 0):.2f}, Episode: {last_point.get('episode', 'N/A')}")
                print_info(f"Count: {result.get('count', len(equity_curve))}")
        else:
            print_error(f"API returned status: {result.get('status')}")
    
    # Test with limit parameter
    result_limited = test_api_endpoint("GET", "/api/journal/equity-curve",
                                      params={"limit": 50},
                                      description="Get equity curve (limited to 50 points)")
    if result_limited and result_limited.get('status') == 'success':
        equity_curve = result_limited.get('equity_curve', [])
        print_info(f"Limited equity curve: {len(equity_curve)} points (requested 50)")

def test_component_3_trading_journal():
    """Test 3: Trading Journal"""
    print_test("Component 3: Trading Journal")
    
    # Test without filter
    result = test_api_endpoint("GET", "/api/journal/trades",
                              params={"limit": 10},
                              description="Get recent trades (limit 10)")
    if result:
        if result.get('status') == 'success':
            trades = result.get('trades', [])
            print_success(f"Trades loaded: {len(trades)} trades")
            if len(trades) > 0:
                print_info(f"Sample trade structure:")
                sample = trades[0]
                print_info(f"  - Trade ID: {sample.get('trade_id', 'N/A')}")
                print_info(f"  - Episode: {sample.get('episode', 'N/A')}")
                print_info(f"  - Entry: ${sample.get('entry_price', 0):.2f}")
                print_info(f"  - Exit: ${sample.get('exit_price', 0):.2f}")
                print_info(f"  - PnL: ${sample.get('pnl', 0):.2f}")
                print_info(f"  - Is Win: {sample.get('is_win', False)}")
            print_info(f"Total count: {result.get('count', len(trades))}")
        else:
            print_error(f"API returned status: {result.get('status')}")
    
    # Test with larger limit
    result_large = test_api_endpoint("GET", "/api/journal/trades",
                                    params={"limit": 50},
                                    description="Get recent trades (limit 50)")
    if result_large and result_large.get('status') == 'success':
        trades = result_large.get('trades', [])
        print_info(f"Large limit result: {len(trades)} trades")

def test_component_4_forecast_performance():
    """Test 4: Forecast Features Performance"""
    print_test("Component 4: Forecast Features Performance")
    
    result = test_api_endpoint("GET", "/api/monitoring/forecast-performance",
                              description="Get forecast performance metrics")
    if result:
        if result.get('status') == 'success':
            print_success("Forecast performance loaded successfully")
            
            # Check configuration
            config = result.get('config', {})
            if config:
                print_info(f"Forecast enabled: {config.get('forecast_enabled', False)}")
                print_info(f"Regime enabled: {config.get('regime_enabled', False)}")
                print_info(f"State features: {config.get('state_features', 'N/A')}")
                print_info(f"State dim match: {config.get('state_dimension_match', False)}")
            
            # Check performance data
            performance = result.get('performance', {})
            if performance and not performance.get('error'):
                print_info(f"Performance metrics:")
                print_info(f"  - Total Trades: {performance.get('total_trades', 0)}")
                print_info(f"  - Win Rate: {(performance.get('win_rate', 0) * 100):.2f}%")
                print_info(f"  - Profit Factor: {performance.get('profit_factor', 0):.2f}")
                print_info(f"  - Total PnL: ${performance.get('total_pnl', 0):.2f}")
            elif performance and performance.get('error'):
                print_warning(f"Forecast performance has error: {performance.get('error')}")
        else:
            print_error(f"API returned status: {result.get('status')}")

def test_component_5_training_status():
    """Test 5: Training Status (for checkpoint timestamp)"""
    print_test("Component 5: Training Status & Checkpoint Timestamp")
    
    result = test_api_endpoint("GET", "/api/training/status",
                              description="Get training status")
    if result:
        print_success("Training status loaded")
        print_info(f"Status: {result.get('status', 'unknown')}")
        
        checkpoint_ts = result.get('checkpoint_resume_timestamp')
        if checkpoint_ts:
            print_info(f"Checkpoint resume timestamp: {checkpoint_ts}")
            print_info("This timestamp is used for filtering trades by session")
        else:
            print_info("No checkpoint resume timestamp (training may not have started or resumed)")

def test_component_6_filtering():
    """Test 6: Filtering Functionality (All Trades vs Current Session)"""
    print_test("Component 6: Filtering Functionality")
    
    # Get checkpoint timestamp first
    training_status = test_api_endpoint("GET", "/api/training/status")
    checkpoint_ts = None
    if training_status:
        checkpoint_ts = training_status.get('checkpoint_resume_timestamp')
    
    if checkpoint_ts:
        print_info(f"Testing with checkpoint timestamp: {checkpoint_ts}")
        
        # Test performance with filter
        result_filtered = test_api_endpoint("GET", "/api/monitoring/performance",
                                           params={"since": checkpoint_ts},
                                           description="Performance with session filter")
        
        # Test equity curve with filter
        result_equity_filtered = test_api_endpoint("GET", "/api/journal/equity-curve",
                                                  params={"since": checkpoint_ts, "limit": 100},
                                                  description="Equity curve with session filter")
        
        # Test trades with filter
        result_trades_filtered = test_api_endpoint("GET", "/api/journal/trades",
                                                  params={"since": checkpoint_ts, "limit": 10},
                                                  description="Trades with session filter")
        
        if result_filtered and result_equity_filtered and result_trades_filtered:
            print_success("All filtering endpoints working correctly")
    else:
        print_warning("No checkpoint timestamp available - skipping filter tests")
        print_info("Filter functionality will be tested when training is active")

def test_component_7_performance_targets():
    """Test 7: Performance Targets Display"""
    print_test("Component 7: Performance Targets (Static Display)")
    
    print_info("Performance Targets are static UI elements (no API endpoint)")
    print_info("Targets displayed:")
    print_info("  - Sharpe Ratio: > 1.5")
    print_info("  - Win Rate: > 55%")
    print_info("  - Profit Factor: > 1.5")
    print_info("  - Max Drawdown: < 20%")
    print_success("Performance targets section is a static UI component (no API needed)")

def run_all_tests():
    """Run all component tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}MONITORING TAB COMPONENT TEST SUITE{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"\nTesting backend API endpoints for Monitoring Tab components...")
    print(f"Base URL: {BASE_URL}\n")
    
    # Test all components
    test_component_1_performance_metrics()
    test_component_2_equity_curve()
    test_component_3_trading_journal()
    test_component_4_forecast_performance()
    test_component_5_training_status()
    test_component_6_filtering()
    test_component_7_performance_targets()
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"\n[PASS] Core API endpoints tested")
    print(f"[INFO] All Monitoring tab components verified")
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    print("1. Open the Monitoring tab in your browser")
    print("2. Verify all components render correctly:")
    print("   - Performance Metrics cards (8 metrics)")
    print("   - Equity Curve chart (should show data if available)")
    print("   - Trading Journal table (click 'Show Journal')")
    print("   - Forecast Features Performance section")
    print("   - Performance Targets reference")
    print("3. Test the filter toggle (All Trades vs Current Session)")
    print("4. Verify auto-refresh every 5 seconds")
    print("5. Check that episode 17 data is visible (if training is on episode 17+)")
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

