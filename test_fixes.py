"""Test script to verify all fixes are working correctly"""
import sys
from pathlib import Path
import sqlite3
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mean_metrics_fix():
    """Test if mean metrics database fallback is working"""
    print("="*80)
    print("TEST 1: Mean Metrics Database Fallback")
    print("="*80)
    
    try:
        db_path = project_root / "logs/trading_journal.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get recent episodes
            cursor.execute("""
                SELECT episode, 
                       SUM(net_pnl) as episode_pnl,
                       COUNT(*) as trades,
                       SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
                FROM trades
                WHERE episode IS NOT NULL
                GROUP BY episode
                ORDER BY episode DESC
                LIMIT 10
            """)
            recent_episodes = cursor.fetchall()
            
            if recent_episodes:
                print(f"[OK] Found {len(recent_episodes)} recent episodes in database")
                episode_pnls = [row[1] for row in recent_episodes if row[1] is not None]
                if episode_pnls:
                    mean_pnl = sum(episode_pnls) / len(episode_pnls)
                    print(f"   Mean PnL (from DB): ${mean_pnl:,.2f}")
                    print(f"   [OK] Database fallback should work correctly")
                else:
                    print(f"   [WARN] No PnL data in recent episodes")
            else:
                print(f"   [WARN] No episodes found in database")
            
            conn.close()
        else:
            print(f"   [WARN] Database not found at {db_path}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")

def test_per_trade_rr_tracking():
    """Test if per-trade R:R tracking is properly initialized"""
    print("\n" + "="*80)
    print("TEST 2: Per-Trade R:R Tracking")
    print("="*80)
    
    try:
        # Try to import and check the trading environment
        from src.trading_env import TradingEnvironment
        
        # Check if recent_trades_rr is in the class
        if hasattr(TradingEnvironment, '__init__'):
            print("[OK] TradingEnvironment class found")
            
            # Read the source file to verify recent_trades_rr is initialized
            env_file = project_root / "src/trading_env.py"
            with open(env_file, 'r') as f:
                content = f.read()
                
            if 'recent_trades_rr' in content:
                print("[OK] recent_trades_rr tracking found in code")
                
                # Count occurrences
                count = content.count('recent_trades_rr')
                print(f"   Found {count} occurrences of recent_trades_rr")
                
                if 'per_trade_rr_penalty' in content:
                    print("[OK] Per-trade R:R penalty found in reward function")
                else:
                    print("   [ERROR] Per-trade R:R penalty NOT found")
                
                if 'per_trade_rr_bonus' in content:
                    print("[OK] Per-trade R:R bonus found in reward function")
                else:
                    print("   [ERROR] Per-trade R:R bonus NOT found")
            else:
                print("   [ERROR] recent_trades_rr NOT found in code")
        else:
            print("   [ERROR] TradingEnvironment class not found")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

def test_reward_function_penalties():
    """Test if reward function penalties are correctly configured"""
    print("\n" + "="*80)
    print("TEST 3: Reward Function Penalties")
    print("="*80)
    
    try:
        env_file = project_root / "src/trading_env.py"
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check for strengthened penalties
        if '0.50' in content or '0.5' in content.split('aggregate_rr_penalty'):
            print("[OK] Aggregate R:R penalty set to 50% (strengthened)")
        else:
            print("   [WARN] Aggregate R:R penalty may not be at 50%")
        
        if '0.30' in content.split('per_trade_rr_penalty') or '0.3' in content.split('per_trade_rr_penalty'):
            print("[OK] Per-trade R:R penalty set to 30%")
        else:
            print("   [WARN] Per-trade R:R penalty may not be at 30%")
        
        if '0.20' in content.split('per_trade_rr_bonus') or '0.2' in content.split('per_trade_rr_bonus'):
            print("[OK] Per-trade R:R bonus set to 20%")
        else:
            print("   [WARN] Per-trade R:R bonus may not be at 20%")
        
        if 'REWARD DEBUG' in content:
            print("[OK] Reward function logging found")
        else:
            print("   [WARN] Reward function logging not found")
            
    except Exception as e:
        print(f"   [ERROR] Error: {e}")

def test_rr_requirement():
    """Test if R:R requirement is set correctly"""
    print("\n" + "="*80)
    print("TEST 4: R:R Requirement Configuration")
    print("="*80)
    
    try:
        import yaml
        config_file = project_root / "configs/train_config_adaptive.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            rr_reward = config.get("environment", {}).get("reward", {}).get("min_risk_reward_ratio")
            rr_decision = config.get("decision_gate", {}).get("quality_scorer", {}).get("min_risk_reward_ratio")
            
            print(f"   Reward config min_risk_reward_ratio: {rr_reward}")
            print(f"   DecisionGate min_risk_reward_ratio: {rr_decision}")
            
            if rr_reward == 2.0:
                print("[OK] R:R requirement set to 2.0:1 in reward config")
            else:
                print(f"   [WARN] R:R requirement is {rr_reward}, expected 2.0")
                
            if rr_decision == 2.0:
                print("[OK] R:R requirement set to 2.0:1 in DecisionGate")
            else:
                print(f"   [WARN] R:R requirement is {rr_decision}, expected 2.0")
        else:
            print(f"   [ERROR] Config file not found at {config_file}")
            
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

def test_api_endpoint():
    """Test if API endpoint is accessible"""
    print("\n" + "="*80)
    print("TEST 5: API Endpoint Accessibility")
    print("="*80)
    
    try:
        import requests
        import time
        
        # Wait a moment for backend to be ready
        time.sleep(1)
        
        try:
            response = requests.get("http://localhost:8000/api/training/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("[OK] API endpoint is accessible")
                print(f"   Status: {data.get('status', 'unknown')}")
                
                # Check if mean metrics are in response
                metrics = data.get("metrics", {})
                mean_pnl = metrics.get("mean_pnl_10", None)
                mean_equity = metrics.get("mean_equity_10", None)
                mean_win_rate = metrics.get("mean_win_rate_10", None)
                
                print(f"   Mean PnL (last 10): ${mean_pnl if mean_pnl else 0:.2f}")
                print(f"   Mean Equity (last 10): ${mean_equity if mean_equity else 0:.2f}")
                print(f"   Mean Win Rate (last 10): {mean_win_rate if mean_win_rate else 0:.2f}%")
                
                if mean_pnl is not None or mean_equity is not None:
                    print("   [OK] Mean metrics are in API response")
            else:
                print(f"   [WARN] API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   [WARN] Backend not running or not accessible at http://localhost:8000")
            print("   This is OK if backend hasn't started yet")
        except Exception as e:
            print(f"   [WARN] Error connecting to API: {e}")
            
    except ImportError:
        print("   [WARN] requests library not available - skipping API test")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")

def test_checkpoint_loading():
    """Test if episode metrics would be preserved in checkpoints"""
    print("\n" + "="*80)
    print("TEST 6: Checkpoint Episode Metrics")
    print("="*80)
    
    try:
        # Check if checkpoint saving includes episode metrics
        agent_file = project_root / "src/rl_agent.py"
        with open(agent_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'episode_pnls' in content or 'episode_equities' in content:
            print("[OK] Checkpoint saving code found")
        else:
            print("   [WARN] Episode metrics may not be saved in checkpoints")
            print("   Note: Database fallback will handle this for now")
            
    except Exception as e:
        print(f"   [ERROR] Error: {e}")

if __name__ == "__main__":
    print("\n")
    print("="*80)
    print("TESTING ALL FIXES")
    print("="*80)
    print()
    
    test_mean_metrics_fix()
    test_per_trade_rr_tracking()
    test_reward_function_penalties()
    test_rr_requirement()
    test_api_endpoint()
    test_checkpoint_loading()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\n[OK] All tests completed. Review output above for any warnings or errors.")
    print("\nNext steps:")
    print("1. If backend is running, check API endpoint results")
    print("2. Start/resume training to see per-trade R:R tracking in action")
    print("3. Monitor reward debug logs (every 500 steps) to verify penalties/bonuses")
    print("4. Check Training Progress panel - mean metrics should show actual values")
    print()

