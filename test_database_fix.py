"""
Test script to verify database initialization and fixes before restarting training.

This script tests:
1. Database initialization
2. Table creation
3. API endpoint database access
4. project_root variable accessibility
"""

import sys
from pathlib import Path
import sqlite3

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_project_root():
    """Test that project_root is accessible"""
    print("\n[TEST 1] Testing project_root variable...")
    try:
        # Simulate the same pattern used in api_server.py
        from src.api_server import project_root as api_project_root
        print(f"  [OK] project_root accessible: {api_project_root}")
        print(f"  [OK] project_root type: {type(api_project_root)}")
        print(f"  [OK] project_root exists: {api_project_root.exists()}")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to access project_root: {e}")
        return False

def test_trading_journal_initialization():
    """Test that TradingJournal initializes database correctly"""
    print("\n[TEST 2] Testing TradingJournal database initialization...")
    try:
        from src.trading_journal import TradingJournal
        
        # Initialize journal (should create database and tables)
        journal = TradingJournal()
        db_path = journal.db_path
        
        print(f"  [OK] TradingJournal created successfully")
        print(f"  [OK] Database path: {db_path}")
        print(f"  [OK] Database file exists: {db_path.exists()}")
        
        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('trades', 'equity_curve', 'episodes')
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}
        required_tables = {'trades', 'equity_curve', 'episodes'}
        
        print(f"  [OK] Required tables: {required_tables}")
        print(f"  [OK] Existing tables: {existing_tables}")
        
        missing_tables = required_tables - existing_tables
        if missing_tables:
            print(f"  [FAIL] Missing tables: {missing_tables}")
            conn.close()
            return False
        else:
            print(f"  [OK] All required tables exist")
        
        # Verify table schemas
        cursor.execute("PRAGMA table_info(trades)")
        trades_columns = [row[1] for row in cursor.fetchall()]
        print(f"  [OK] trades table columns: {len(trades_columns)} columns")
        
        cursor.execute("PRAGMA table_info(equity_curve)")
        equity_columns = [row[1] for row in cursor.fetchall()]
        print(f"  [OK] equity_curve table columns: {len(equity_columns)} columns")
        
        cursor.execute("PRAGMA table_info(episodes)")
        episodes_columns = [row[1] for row in cursor.fetchall()]
        print(f"  [OK] episodes table columns: {len(episodes_columns)} columns")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] Failed to initialize TradingJournal: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_queries():
    """Test that database queries work without errors"""
    print("\n[TEST 3] Testing database queries...")
    try:
        from src.trading_journal import TradingJournal
        
        journal = TradingJournal()
        db_path = journal.db_path
        
        if not db_path.exists():
            print(f"  [FAIL] Database file does not exist: {db_path}")
            return False
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Test query that was failing: SELECT from trades
        try:
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            print(f"  [OK] Query 'SELECT COUNT(*) FROM trades' succeeded: {count} rows")
        except Exception as e:
            print(f"  [FAIL] Query 'SELECT COUNT(*) FROM trades' failed: {e}")
            conn.close()
            return False
        
        # Test query that was failing: SELECT from equity_curve
        try:
            cursor.execute("SELECT COUNT(*) FROM equity_curve")
            count = cursor.fetchone()[0]
            print(f"  [OK] Query 'SELECT COUNT(*) FROM equity_curve' succeeded: {count} rows")
        except Exception as e:
            print(f"  [FAIL] Query 'SELECT COUNT(*) FROM equity_curve' failed: {e}")
            conn.close()
            return False
        
        # Test query with WHERE clause (like in training_status)
        try:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", ("2024-01-01",))
            count = cursor.fetchone()[0]
            print(f"  [OK] Query with WHERE clause succeeded: {count} rows")
        except Exception as e:
            print(f"  [FAIL] Query with WHERE clause failed: {e}")
            conn.close()
            return False
        
        # Test query with is_win filter (like in training_status)
        try:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 1")
            count = cursor.fetchone()[0]
            print(f"  [OK] Query with is_win filter succeeded: {count} rows")
        except Exception as e:
            print(f"  [FAIL] Query with is_win filter failed: {e}")
            conn.close()
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] Database query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_journal_methods():
    """Test TradingJournal methods that are used by API endpoints"""
    print("\n[TEST 4] Testing TradingJournal methods...")
    try:
        from src.trading_journal import TradingJournal, get_journal
        
        # Test get_journal() (used by equity-curve endpoint)
        journal = get_journal()
        print(f"  [OK] get_journal() succeeded")
        
        # Test get_trades() method
        trades = journal.get_trades(limit=10)
        print(f"  [OK] get_trades() succeeded: {len(trades)} trades")
        
        # Test get_equity_curve() method
        curve = journal.get_equity_curve(limit=100)
        print(f"  [OK] get_equity_curve() succeeded: {len(curve)} points")
        
        # Test get_statistics() method
        stats = journal.get_statistics()
        print(f"  [OK] get_statistics() succeeded")
        print(f"     - Total trades: {stats.get('total_trades', 0)}")
        print(f"     - Win rate: {stats.get('win_rate', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] TradingJournal methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_server_imports():
    """Test that api_server can import and access project_root"""
    print("\n[TEST 5] Testing API server imports and project_root access...")
    try:
        # Import the module (this will execute the top-level code)
        import src.api_server
        
        # Check that project_root is defined
        if hasattr(src.api_server, 'project_root'):
            project_root = src.api_server.project_root
            print(f"  [OK] project_root defined in api_server: {project_root}")
            print(f"  [OK] project_root type: {type(project_root)}")
            print(f"  [OK] project_root exists: {project_root.exists()}")
            return True
        else:
            print(f"  [FAIL] project_root not found in api_server module")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Failed to import api_server: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulated_training_status_access():
    """Simulate the database access pattern used in training_status endpoint"""
    print("\n[TEST 6] Testing simulated training_status database access...")
    try:
        from pathlib import Path
        import sqlite3
        from src.trading_journal import TradingJournal
        
        # This simulates the code in training_status() function
        # Use global project_root (not local assignment)
        project_root = Path(__file__).parent  # This would be the actual project root
        
        # Ensure database is initialized (like we added in the fix)
        journal = TradingJournal()
        db_path = project_root / "logs/trading_journal.db"
        
        if not db_path.exists():
            print(f"  [FAIL] Database file does not exist: {db_path}")
            return False
        
        # Simulate the query from training_status
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # This is the exact query pattern used in training_status
        training_start_ts = "2024-01-01T00:00:00"
        cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (training_start_ts,))
        db_total_trades = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 1 AND timestamp >= ?", (training_start_ts,))
        db_winning_trades = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win = 0 AND timestamp >= ?", (training_start_ts,))
        db_losing_trades = cursor.fetchone()[0] or 0
        
        print(f"  [OK] Simulated training_status queries succeeded")
        print(f"     - Total trades: {db_total_trades}")
        print(f"     - Winning trades: {db_winning_trades}")
        print(f"     - Losing trades: {db_losing_trades}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] Simulated training_status access failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("DATABASE FIX VERIFICATION TESTS")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("project_root access", test_project_root()))
    results.append(("TradingJournal initialization", test_trading_journal_initialization()))
    results.append(("Database queries", test_database_queries()))
    results.append(("Journal methods", test_journal_methods()))
    results.append(("API server imports", test_api_server_imports()))
    results.append(("Simulated training_status", test_simulated_training_status_access()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  [SUCCESS] ALL TESTS PASSED! Database fixes are working correctly.")
        print("  [OK] Safe to restart training.")
        return 0
    else:
        print("\n  [WARNING] SOME TESTS FAILED! Please review the errors above.")
        print("  [FAIL] Do not restart training until issues are resolved.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

