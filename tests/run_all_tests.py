"""
Run All Automated Tests

Runs all test suites for:
- Phase 1.4: Regime Features
- Phase 3.3: Stop-Loss Logic
- Phase 3.4: Time-of-Day Filter
"""

import sys
from pathlib import Path

# Configure stdout for Windows Unicode support (only once)
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

def run_test_suite(module_name, test_file):
    """Run a test suite and return results"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {module_name}")
    print("=" * 80)
    
    try:
        # Import and run the test module
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run the tests
        if hasattr(module, 'run_all_tests'):
            return module.run_all_tests()
        else:
            print(f"  ⚠️  Module {module_name} doesn't have run_all_tests() function")
            return False
    except Exception as e:
        print(f"  ❌ Failed to run {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test suites"""
    print("=" * 80)
    print("AUTOMATED TEST SUITE")
    print("Testing All Implemented Features")
    print("=" * 80)
    
    test_suites = [
        ("Regime Features (Phase 1.4)", "tests/test_regime_features.py"),
        ("Stop-Loss Logic (Phase 3.3)", "tests/test_stop_loss.py"),
        ("Time-of-Day Filter (Phase 3.4)", "tests/test_time_filter.py"),
        ("Forecast Features (Phase 4.1)", "tests/test_forecast_features.py"),
    ]
    
    results = {}
    for name, test_file in test_suites:
        test_path = project_root / test_file
        if test_path.exists():
            results[name] = run_test_suite(name, test_path)
        else:
            print(f"\n[WARN] Test file not found: {test_file}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TEST SUITES PASSED")
        print("[SUCCESS] Ready to proceed with training")
        return True
    else:
        print(f"\n[WARN] {total - passed} test suite(s) failed")
        print("[WARN] Review failures before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

