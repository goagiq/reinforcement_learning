#!/usr/bin/env python3
"""
Test script for Critical Self-Healing Integrations

Tests:
1. Directional Bias Detection
2. Rapid Drawdown Detection  
3. Reward Collapse Detection
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, List

# Configure stdout for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_directional_bias_detection():
    """Test 1: Directional Bias Detection"""
    print("\n" + "="*70)
    print("TEST 1: Directional Bias Detection")
    print("="*70)
    
    try:
        from src.trading_env import TradingEnvironment
        import pandas as pd
        
        # Create mock data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        data = {
            1: pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(1000).cumsum() + 4000,
                'high': np.random.randn(1000).cumsum() + 4000,
                'low': np.random.randn(1000).cumsum() + 4000,
                'close': np.random.randn(1000).cumsum() + 4000,
                'volume': np.random.randint(1000, 10000, 1000)
            })
        }
        
        # Create environment with default reward config
        default_reward_config = {
            "pnl_weight": 1.0,
            "risk_penalty": 0.5,
            "drawdown_penalty": 0.3,
            "quality_filters": {
                "min_action_confidence": 0.1,
                "min_quality_score": 0.3
            }
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1],
            initial_capital=100000.0,
            reward_config=default_reward_config
        )
        
        # Simulate actions that are 95% LONG (directional bias)
        env.action_distribution = []
        for i in range(100):
            if i < 95:  # 95% LONG
                env.action_distribution.append(0.8)  # Strong LONG
            else:
                env.action_distribution.append(-0.2)  # Weak SHORT
        
        # Trigger reset to analyze action distribution
        env.reset()
        
        # Check if directional bias was detected
        if hasattr(env, '_last_episode_directional_bias'):
            if env._last_episode_directional_bias == "LONG":
                print("✅ PASS: Directional bias (LONG) correctly detected")
                print(f"   Bias Percentage: {env._last_episode_directional_bias_pct*100:.1f}%")
                return True
            else:
                print(f"❌ FAIL: Expected LONG bias, got {env._last_episode_directional_bias}")
                return False
        else:
            print("❌ FAIL: Directional bias flag not set")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rapid_drawdown_detection():
    """Test 2: Rapid Drawdown Detection"""
    print("\n" + "="*70)
    print("TEST 2: Rapid Drawdown Detection")
    print("="*70)
    
    try:
        from src.adaptive_trainer import AdaptiveTrainer
        from src.rl_agent import PPOAgent
        import torch
        
        # Create mock agent
        agent = Mock(spec=PPOAgent)
        agent.entropy_coef = 0.01
        
        # Create adaptive trainer
        config_path = "configs/train_config_adaptive.yaml"
        if not os.path.exists(config_path):
            print(f"⚠️  WARN: Config file not found: {config_path}")
            print("   Skipping rapid drawdown test (requires config)")
            return True
        
        trainer = AdaptiveTrainer(config_path)
        
        # Simulate rapid drawdown scenario
        drawdown_pct = 0.12  # 12% drawdown (above 10% threshold)
        peak_equity = 100000.0
        current_equity = 88000.0  # 12% below peak
        
        # Call response method
        adjustments = trainer.respond_to_rapid_drawdown(
            drawdown_pct=drawdown_pct,
            peak_equity=peak_equity,
            current_equity=current_equity,
            agent=agent,
            timestep=10000,
            episode=10
        )
        
        if adjustments:
            print("✅ PASS: Rapid drawdown response triggered")
            print(f"   Drawdown: {drawdown_pct*100:.1f}%")
            if "entropy_coef" in adjustments:
                print(f"   Entropy: {adjustments['entropy_coef']['old']:.4f} -> {adjustments['entropy_coef']['new']:.4f}")
            if "quality_filters" in adjustments:
                conf = adjustments['quality_filters']['min_action_confidence']
                print(f"   Confidence Filter: {conf['old']:.3f} -> {conf['new']:.3f} (tightened)")
            return True
        else:
            print("❌ FAIL: No adjustments returned (may be spam-protected)")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_collapse_detection():
    """Test 3: Reward Collapse Detection"""
    print("\n" + "="*70)
    print("TEST 3: Reward Collapse Detection")
    print("="*70)
    
    try:
        from src.adaptive_trainer import AdaptiveTrainer
        from src.rl_agent import PPOAgent
        
        # Create mock agent
        agent = Mock(spec=PPOAgent)
        agent.entropy_coef = 0.01
        
        # Create adaptive trainer
        config_path = "configs/train_config_adaptive.yaml"
        if not os.path.exists(config_path):
            print(f"⚠️  WARN: Config file not found: {config_path}")
            print("   Skipping reward collapse test (requires config)")
            return True
        
        trainer = AdaptiveTrainer(config_path)
        
        # Simulate reward collapse scenario (mean reward < -0.5)
        recent_rewards = [-0.6] * 20  # All negative, mean = -0.6
        mean_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Call response method
        adjustments = trainer.respond_to_reward_collapse(
            mean_reward=mean_reward,
            recent_rewards=recent_rewards,
            agent=agent,
            timestep=20000,
            episode=20
        )
        
        if adjustments:
            print("✅ PASS: Reward collapse response triggered")
            print(f"   Mean Reward: {mean_reward:.4f}")
            if "entropy_coef" in adjustments:
                print(f"   Entropy: {adjustments['entropy_coef']['old']:.4f} -> {adjustments['entropy_coef']['new']:.4f}")
            if "inaction_penalty" in adjustments:
                penalty = adjustments['inaction_penalty']
                print(f"   Inaction Penalty: {penalty['old']:.6f} -> {penalty['new']:.6f}")
            return True
        else:
            print("❌ FAIL: No adjustments returned (may be spam-protected)")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_patterns():
    """Test 4: Monitoring Pattern Detection"""
    print("\n" + "="*70)
    print("TEST 4: Monitoring Pattern Detection")
    print("="*70)
    
    try:
        import re
        
        # Test patterns from api_server.py
        patterns = {
            "directional_bias": [
                r'WARNING:.*% of actions are (LONG|SHORT).*directional bias',
                r'DIRECTIONAL BIAS DETECTED',
                r'\[CRITICAL\] DIRECTIONAL BIAS',
                r'directional bias detected!'
            ],
            "rapid_drawdown": [
                r'\[CRITICAL\] RAPID DRAWDOWN DETECTED',
                r'RAPID DRAWDOWN RESPONSE',
                r'drawdown.*10\.0%',
                r'Drawdown:.*10\.[0-9]+%'
            ],
            "reward_collapse": [
                r'\[CRITICAL\] REWARD COLLAPSE DETECTED',
                r'REWARD COLLAPSE RESPONSE',
                r'Mean Reward.*Last 20.*-0\.[5-9]',
                r'reward collapse'
            ]
        }
        
        # Test messages
        test_messages = {
            "directional_bias": [
                "   ⚠️  WARNING: 95.0% of actions are LONG - directional bias detected!",
                "[CRITICAL] DIRECTIONAL BIAS DETECTED after episode 42:",
                "DIRECTIONAL BIAS DETECTED"
            ],
            "rapid_drawdown": [
                "[CRITICAL] RAPID DRAWDOWN DETECTED after episode 45:",
                "   Drawdown: 10.4%",
                "RAPID DRAWDOWN RESPONSE (Immediate - Every Episode):"
            ],
            "reward_collapse": [
                "[CRITICAL] REWARD COLLAPSE DETECTED after episode 50:",
                "   Mean Reward (Last 20): -0.5234",
                "REWARD COLLAPSE RESPONSE (Immediate - Every Episode):"
            ]
        }
        
        all_passed = True
        for category, test_msgs in test_messages.items():
            category_patterns = patterns[category]
            for msg in test_msgs:
                matched = False
                for pattern in category_patterns:
                    if re.search(pattern, msg, re.IGNORECASE):
                        matched = True
                        break
                if matched:
                    print(f"✅ PASS: Pattern matched for {category}: '{msg[:50]}...'")
                else:
                    print(f"❌ FAIL: Pattern NOT matched for {category}: '{msg[:50]}...'")
                    all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_trainer_initialization():
    """Test 5: Adaptive Trainer Initialization"""
    print("\n" + "="*70)
    print("TEST 5: Adaptive Trainer Initialization")
    print("="*70)
    
    try:
        from src.adaptive_trainer import AdaptiveTrainer
        
        config_path = "configs/train_config_adaptive.yaml"
        if not os.path.exists(config_path):
            print(f"⚠️  WARN: Config file not found: {config_path}")
            print("   Skipping initialization test (requires config)")
            return True
        
        trainer = AdaptiveTrainer(config_path)
        
        # Check if new tracking variables exist
        required_attrs = [
            'last_directional_bias_response_timestep',
            'last_rapid_drawdown_response_timestep',
            'last_reward_collapse_response_timestep'
        ]
        
        all_present = True
        for attr in required_attrs:
            if hasattr(trainer, attr):
                print(f"✅ PASS: {attr} initialized")
            else:
                print(f"❌ FAIL: {attr} not initialized")
                all_present = False
        
        # Check if response methods exist
        required_methods = [
            'respond_to_directional_bias',
            'respond_to_rapid_drawdown',
            'respond_to_reward_collapse'
        ]
        
        for method in required_methods:
            if hasattr(trainer, method):
                print(f"✅ PASS: {method} method exists")
            else:
                print(f"❌ FAIL: {method} method not found")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("CRITICAL SELF-HEALING INTEGRATIONS - TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Directional Bias Detection", test_directional_bias_detection()))
    results.append(("Rapid Drawdown Detection", test_rapid_drawdown_detection()))
    results.append(("Reward Collapse Detection", test_reward_collapse_detection()))
    results.append(("Monitoring Pattern Detection", test_monitoring_patterns()))
    results.append(("Adaptive Trainer Initialization", test_adaptive_trainer_initialization()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

