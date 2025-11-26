"""
Test Time-of-Day Filter (Phase 3.4)

Tests:
- Time filter initialization
- Avoid period detection
- Strict vs lenient modes
- Timezone handling
"""

import sys
from pathlib import Path
from datetime import datetime

# Configure stdout for Windows Unicode support (only if not already wrapped)
if sys.platform == 'win32':
    import io
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass  # Already wrapped or can't wrap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.time_of_day_filter import TimeOfDayFilter


def test_time_filter_initialization():
    """Test time filter can be initialized"""
    print("\n[TEST] Time Filter Initialization...")
    try:
        filter = TimeOfDayFilter({
            "enabled": True,
            "timezone": "America/New_York",
            "avoid_hours": [(11, 30, 14, 0)],
            "strict_mode": False
        })
        
        assert filter is not None
        assert filter.enabled is True
        assert filter.strict_mode is False
        assert len(filter.avoid_periods) == 1
        
        print("  [OK] Time filter initialized successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_avoid_period_detection():
    """Test that avoid periods are detected correctly"""
    print("\n[TEST] Avoid Period Detection...")
    try:
        filter = TimeOfDayFilter({
            "enabled": True,
            "timezone": "America/New_York",
            "avoid_hours": [(11, 30, 14, 0)],  # 11:30-14:00
            "strict_mode": False
        })
        
        # Test during avoid period (12:00)
        dt_avoid = datetime(2024, 1, 1, 12, 0)
        assert filter.is_in_avoid_period(dt_avoid), "12:00 should be in avoid period"
        
        # Test outside avoid period (10:00)
        dt_allowed = datetime(2024, 1, 1, 10, 0)
        assert not filter.is_in_avoid_period(dt_allowed), "10:00 should NOT be in avoid period"
        
        # Test at boundary (11:30)
        dt_boundary_start = datetime(2024, 1, 1, 11, 30)
        assert filter.is_in_avoid_period(dt_boundary_start), "11:30 should be in avoid period"
        
        # Test at boundary (14:00)
        dt_boundary_end = datetime(2024, 1, 1, 14, 0)
        assert filter.is_in_avoid_period(dt_boundary_end), "14:00 should be in avoid period"
        
        print("  [OK] Avoid period detection works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strict_mode():
    """Test strict mode (reject trades)"""
    print("\n[TEST] Strict Mode...")
    try:
        filter = TimeOfDayFilter({
            "enabled": True,
            "timezone": "America/New_York",
            "avoid_hours": [(11, 30, 14, 0)],
            "strict_mode": True  # Reject trades
        })
        
        # Test during avoid period
        dt_avoid = datetime(2024, 1, 1, 12, 0)
        action, confidence, reason = filter.filter_decision(dt_avoid, 0.5, 0.8)
        
        assert action == 0.0, "Action should be 0.0 in strict mode"
        assert confidence == 0.0, "Confidence should be 0.0 in strict mode"
        assert reason == "rejected_time_of_day", "Reason should be 'rejected_time_of_day'"
        
        print("  [OK] Strict mode rejects trades correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lenient_mode():
    """Test lenient mode (reduce confidence)"""
    print("\n[TEST] Lenient Mode...")
    try:
        filter = TimeOfDayFilter({
            "enabled": True,
            "timezone": "America/New_York",
            "avoid_hours": [(11, 30, 14, 0)],
            "strict_mode": False,  # Reduce confidence
            "confidence_reduction": 0.3  # 30% reduction
        })
        
        # Test during avoid period
        dt_avoid = datetime(2024, 1, 1, 12, 0)
        action, confidence, reason = filter.filter_decision(dt_avoid, 0.5, 0.8)
        
        assert action == 0.5, "Action should be unchanged in lenient mode"
        assert confidence < 0.8, "Confidence should be reduced"
        assert confidence == 0.8 * 0.7, f"Expected confidence 0.56, got {confidence}"  # 0.8 * (1 - 0.3)
        assert reason == "reduced_confidence_time_of_day", "Reason should be 'reduced_confidence_time_of_day'"
        
        # Test outside avoid period
        dt_allowed = datetime(2024, 1, 1, 10, 0)
        action, confidence, reason = filter.filter_decision(dt_allowed, 0.5, 0.8)
        
        assert action == 0.5, "Action should be unchanged"
        assert confidence == 0.8, "Confidence should be unchanged"
        assert reason == "allowed", "Reason should be 'allowed'"
        
        print("  [OK] Lenient mode reduces confidence correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disabled_filter():
    """Test that disabled filter doesn't filter anything"""
    print("\n[TEST] Disabled Filter...")
    try:
        filter = TimeOfDayFilter({
            "enabled": False,
            "timezone": "America/New_York",
            "avoid_hours": [(11, 30, 14, 0)],
            "strict_mode": True
        })
        
        # Even during avoid period, should allow
        dt_avoid = datetime(2024, 1, 1, 12, 0)
        action, confidence, reason = filter.filter_decision(dt_avoid, 0.5, 0.8)
        
        assert action == 0.5, "Action should be unchanged when disabled"
        assert confidence == 0.8, "Confidence should be unchanged when disabled"
        assert reason == "time_filter_disabled", "Reason should be 'time_filter_disabled'"
        
        print("  [OK] Disabled filter doesn't filter trades")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_avoid_periods():
    """Test multiple avoid periods"""
    print("\n[TEST] Multiple Avoid Periods...")
    try:
        filter = TimeOfDayFilter({
            "enabled": True,
            "timezone": "America/New_York",
            "avoid_hours": [
                (11, 30, 14, 0),  # Lunch
                (15, 30, 16, 0),  # Close
            ],
            "strict_mode": False
        })
        
        # Test first avoid period
        dt_lunch = datetime(2024, 1, 1, 12, 0)
        assert filter.is_in_avoid_period(dt_lunch), "Should be in lunch period"
        
        # Test second avoid period
        dt_close = datetime(2024, 1, 1, 15, 45)
        assert filter.is_in_avoid_period(dt_close), "Should be in close period"
        
        # Test outside both periods
        dt_allowed = datetime(2024, 1, 1, 10, 0)
        assert not filter.is_in_avoid_period(dt_allowed), "Should NOT be in any avoid period"
        
        print("  [OK] Multiple avoid periods work correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all time filter tests"""
    print("=" * 80)
    print("TESTING TIME-OF-DAY FILTER (Phase 3.4)")
    print("=" * 80)
    
    tests = [
        ("Time Filter Initialization", test_time_filter_initialization),
        ("Avoid Period Detection", test_avoid_period_detection),
        ("Strict Mode", test_strict_mode),
        ("Lenient Mode", test_lenient_mode),
        ("Disabled Filter", test_disabled_filter),
        ("Multiple Avoid Periods", test_multiple_avoid_periods),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED - Time filter ready for use")
        return True
    else:
        print(f"\n[WARN] {total - passed} test(s) failed - Review time filter implementation")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

