"""
Diagnostic script to understand model behavior during evaluation.
Logs detailed information about actions, filters, and trades.
"""

import sys
from pathlib import Path
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from src.trading_hours import TradingHoursManager

def diagnose_model(model_path: str, config_path: str = "configs/train_config_adaptive.yaml", n_episodes: int = 3):
    """Diagnose model behavior with detailed logging"""
    print("="*80)
    print("MODEL BEHAVIOR DIAGNOSTIC")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("\nLoading data...")
    extractor = DataExtractor()
    instrument = config["environment"]["instrument"]
    timeframes = config["environment"]["timeframes"]
    trading_hours_cfg = config["environment"].get("trading_hours", {})
    trading_hours_manager = None
    if trading_hours_cfg.get("enabled"):
        trading_hours_manager = TradingHoursManager.from_dict(trading_hours_cfg)
    
    multi_tf_data = extractor.load_multi_timeframe_data(
        instrument,
        timeframes,
        trading_hours=trading_hours_manager
    )
    print("Data loaded successfully")
    
    # Create environment with ALL parameters from config
    print("\nCreating environment...")
    action_threshold = config["environment"].get("action_threshold", 0.05)
    max_episode_steps = config["environment"].get("max_episode_steps", 10000)
    
    print(f"  Action threshold: {action_threshold}")
    print(f"  Max episode steps: {max_episode_steps}")
    
    env = TradingEnvironment(
        data=multi_tf_data,
        timeframes=config["environment"]["timeframes"],
        initial_capital=config["risk_management"]["initial_capital"],
        transaction_cost=config["risk_management"]["commission"] / config["risk_management"]["initial_capital"],
        reward_config=config["environment"]["reward"],
        action_threshold=action_threshold,
        max_episode_steps=max_episode_steps
    )
    
    # Load agent
    print(f"\nLoading agent from: {model_path}")
    
    # Load agent - try normal load first, then transfer learning if needed
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_range=tuple(config["environment"]["action_range"]),
        device="cpu"
    )
    
    try:
        # Try normal loading first
        agent.load(model_path)
        print("Agent loaded successfully")
    except Exception as e:
        error_msg = str(e)
        # Check if it's an architecture mismatch error
        if "size mismatch" in error_msg or "hidden_dims" in error_msg.lower():
            print(f"\nArchitecture mismatch detected!")
            print(f"  Attempting to use transfer learning...")
            
            try:
                # Use transfer learning
                transfer_strategy = config.get("training", {}).get("transfer_strategy", "copy_and_extend")
                # Temporarily redirect stdout to avoid Unicode issues
                import sys
                import io
                import re
                
                # Create a filter that removes problematic Unicode
                class UnicodeFilter:
                    def __init__(self, stream):
                        self.stream = stream
                    def write(self, text):
                        # Remove emojis and other problematic Unicode
                        text = re.sub(r'[^\x00-\x7F]+', '', text)
                        self.stream.write(text)
                    def flush(self):
                        self.stream.flush()
                
                old_stdout = sys.stdout
                sys.stdout = UnicodeFilter(old_stdout)
                try:
                    agent.load_with_transfer(model_path, transfer_strategy=transfer_strategy)
                finally:
                    sys.stdout = old_stdout
                print("Agent loaded successfully with transfer learning")
            except Exception as e2:
                print(f"Error with transfer learning: {e2}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"Error loading agent: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    agent.actor.eval()
    agent.critic.eval()
    
    # Run diagnostic episodes
    print(f"\nRunning diagnostic evaluation over {n_episodes} episodes...")
    print("="*80)
    
    for episode in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*80}")
        
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        action_count = 0
        trade_count = 0
        rejected_count = 0
        
        action_history = []
        trade_history = []
        
        while not done and episode_step < 100:  # Limit to first 100 steps for diagnosis
            # Get action
            action, value, log_prob = agent.select_action(state, deterministic=True)
            action_value = float(action[0])
            action_count += 1
            
            # Log action details
            if episode_step < 10 or abs(action_value) > 0.01:  # Log first 10 steps or significant actions
                print(f"\nStep {episode_step}:")
                print(f"  Action value: {action_value:.6f}")
                print(f"  Action abs: {abs(action_value):.6f}")
                print(f"  Action threshold: {action_threshold}")
                print(f"  Would trigger trade: {abs(action_value) > action_threshold}")
            
            # Step environment
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Check if trade occurred
            if isinstance(step_info, dict):
                trades = step_info.get("trades", 0)
                if trades > trade_count:
                    trade_count = trades
                    trade_history.append({
                        "step": episode_step,
                        "action": action_value,
                        "reward": reward,
                        "pnl": step_info.get("pnl", 0),
                        "win_rate": step_info.get("win_rate", 0)
                    })
                    print(f"  >>> TRADE EXECUTED at step {episode_step}")
                    print(f"      Action: {action_value:.6f}")
                    print(f"      Reward: {reward:.6f}")
                    print(f"      PnL: ${step_info.get('pnl', 0):.2f}")
                    print(f"      Win Rate: {step_info.get('win_rate', 0)*100:.1f}%")
            
            # Track position changes
            if abs(action_value) > action_threshold:
                if episode_step < 10:
                    print(f"  Action above threshold, but no trade logged")
            
            episode_step += 1
        
        # Episode summary
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*80}")
        print(f"Total steps: {episode_step}")
        print(f"Total actions: {action_count}")
        print(f"Trades executed: {trade_count}")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Final PnL: ${step_info.get('pnl', 0) if isinstance(step_info, dict) else 0:.2f}")
        print(f"Final win rate: {step_info.get('win_rate', 0)*100 if isinstance(step_info, dict) else 0:.1f}%")
        
        if trade_history:
            print(f"\nTrade History:")
            for i, trade in enumerate(trade_history, 1):
                print(f"  Trade {i}: Step {trade['step']}, Action: {trade['action']:.6f}, "
                      f"PnL: ${trade['pnl']:.2f}, Win Rate: {trade['win_rate']*100:.1f}%")
        else:
            print(f"\nNo trades executed in this episode")
        
        # Analyze action distribution
        if action_history:
            actions = [a for a in action_history]
            print(f"\nAction Statistics:")
            print(f"  Mean: {np.mean(actions):.6f}")
            print(f"  Std: {np.std(actions):.6f}")
            print(f"  Min: {np.min(actions):.6f}")
            print(f"  Max: {np.max(actions):.6f}")
            print(f"  Actions > threshold ({action_threshold}): {sum(1 for a in actions if abs(a) > action_threshold)}")
            print(f"  Actions > 0.01: {sum(1 for a in actions if abs(a) > 0.01)}")
            print(f"  Actions > 0.1: {sum(1 for a in actions if abs(a) > 0.1)}")
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Model path")
    parser.add_argument("--config", type=str, default="configs/train_config_adaptive.yaml", help="Config path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    args = parser.parse_args()
    
    diagnose_model(args.model, args.config, args.episodes)

