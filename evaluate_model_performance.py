"""
Evaluate Trained Model Performance
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from src.trading_hours import TradingHoursManager

def evaluate_model(model_path: str, config_path: str = "configs/train_config_adaptive.yaml", n_episodes: int = 20):
    """Evaluate model performance"""
    print("="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Episodes: {n_episodes}")
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
    
    # Create environment
    print("\nCreating environment...")
    env = TradingEnvironment(
        data=multi_tf_data,
        timeframes=config["environment"]["timeframes"],
        initial_capital=config["risk_management"]["initial_capital"],
        transaction_cost=config["risk_management"]["commission"] / config["risk_management"]["initial_capital"],
        reward_config=config["environment"]["reward"],
        action_threshold=config["environment"].get("action_threshold", 0.05),
        max_episode_steps=config["environment"].get("max_episode_steps", 10000)
    )
    
    # Load agent
    print(f"\nLoading agent from: {model_path}")
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_range=tuple(config["environment"]["action_range"]),
        device="cpu"
    )
    
    try:
        agent.load(model_path)
        print("Agent loaded successfully")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None
    
    agent.actor.eval()
    agent.critic.eval()
    
    # Run evaluation
    print(f"\nRunning evaluation over {n_episodes} episodes...")
    print("-"*80)
    
    all_rewards = []
    all_pnls = []
    all_trades = []
    all_win_rates = []
    all_max_drawdowns = []
    all_equities = []
    winning_trades = 0
    losing_trades = 0
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_step += 1
        
        # Get final metrics
        final_info = step_info if isinstance(step_info, dict) else {}
        pnl = final_info.get("pnl", 0)
        trades = final_info.get("trades", 0)
        win_rate = final_info.get("win_rate", 0)
        max_dd = final_info.get("max_drawdown", 0)
        equity = final_info.get("equity", config["risk_management"]["initial_capital"])
        
        all_rewards.append(episode_reward)
        all_pnls.append(pnl)
        all_trades.append(trades)
        all_win_rates.append(win_rate)
        all_max_drawdowns.append(max_dd)
        all_equities.append(equity)
        
        # Count wins/losses
        if pnl > 0:
            winning_trades += 1
        elif pnl < 0:
            losing_trades += 1
        
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode {episode+1:3d}/{n_episodes}: "
                  f"Reward: {episode_reward:8.2f}, "
                  f"PnL: ${pnl:10.2f}, "
                  f"Trades: {trades:3d}, "
                  f"Win Rate: {win_rate*100:5.1f}%, "
                  f"Equity: ${equity:10.2f}")
    
    # Calculate aggregate metrics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_pnl = np.mean(all_pnls)
    std_pnl = np.std(all_pnls)
    total_pnl = sum(all_pnls)
    mean_trades = np.mean(all_trades)
    total_trades = sum(all_trades)
    mean_win_rate = np.mean(all_win_rates)
    mean_max_dd = np.mean(all_max_drawdowns)
    max_max_dd = max(all_max_drawdowns)
    mean_equity = np.mean(all_equities)
    final_equity = all_equities[-1]
    initial_capital = config["risk_management"]["initial_capital"]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Calculate win rate from PnL
    profitable_episodes = sum(1 for pnl in all_pnls if pnl > 0)
    losing_episodes = sum(1 for pnl in all_pnls if pnl < 0)
    episode_win_rate = profitable_episodes / len(all_pnls) if all_pnls else 0
    
    # Calculate profit factor
    gross_profit = sum(pnl for pnl in all_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in all_pnls if pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate Sharpe ratio (simplified)
    if len(all_pnls) > 1 and std_pnl > 0:
        sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0
    
    print(f"\nRewards:")
    print(f"  Mean Reward: {mean_reward:10.2f} +/- {std_reward:8.2f}")
    print(f"  Best Reward: {max(all_rewards):10.2f}")
    print(f"  Worst Reward: {min(all_rewards):10.2f}")
    
    print(f"\nFinancial Performance:")
    print(f"  Total PnL: ${total_pnl:10.2f}")
    print(f"  Mean PnL per Episode: ${mean_pnl:10.2f} +/- ${std_pnl:8.2f}")
    print(f"  Best Episode PnL: ${max(all_pnls):10.2f}")
    print(f"  Worst Episode PnL: ${min(all_pnls):10.2f}")
    print(f"  Initial Capital: ${initial_capital:10.2f}")
    print(f"  Final Equity: ${final_equity:10.2f}")
    print(f"  Mean Equity: ${mean_equity:10.2f}")
    print(f"  Total Return: {total_return*100:8.2f}%")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {total_trades:5d}")
    print(f"  Mean Trades per Episode: {mean_trades:5.1f}")
    print(f"  Profitable Episodes: {profitable_episodes:3d} / {len(all_pnls)}")
    print(f"  Losing Episodes: {losing_episodes:3d} / {len(all_pnls)}")
    print(f"  Episode Win Rate: {episode_win_rate*100:6.2f}%")
    print(f"  Mean Win Rate (per episode): {mean_win_rate*100:6.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Mean Max Drawdown: {mean_max_dd*100:8.2f}%")
    print(f"  Worst Max Drawdown: {max_max_dd*100:8.2f}%")
    print(f"  Profit Factor: {profit_factor:8.2f}")
    print(f"  Sharpe Ratio: {sharpe_ratio:8.2f}")
    
    # Assessment
    print(f"\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if total_return > 0:
        print(f"  [OK] Positive total return: {total_return*100:.2f}%")
    else:
        print(f"  [WARN] Negative total return: {total_return*100:.2f}%")
    
    if mean_pnl > 0:
        print(f"  [OK] Positive mean PnL: ${mean_pnl:.2f}")
    else:
        print(f"  [WARN] Negative mean PnL: ${mean_pnl:.2f}")
    
    if episode_win_rate >= 0.5:
        print(f"  [OK] Episode win rate >= 50%: {episode_win_rate*100:.1f}%")
    elif episode_win_rate >= 0.4:
        print(f"  [WARN] Episode win rate < 50%: {episode_win_rate*100:.1f}%")
    else:
        print(f"  [CRITICAL] Episode win rate < 40%: {episode_win_rate*100:.1f}%")
    
    if profit_factor >= 1.5:
        print(f"  [OK] Profit factor >= 1.5: {profit_factor:.2f}")
    elif profit_factor >= 1.0:
        print(f"  [WARN] Profit factor < 1.5: {profit_factor:.2f}")
    else:
        print(f"  [CRITICAL] Profit factor < 1.0: {profit_factor:.2f}")
    
    if sharpe_ratio > 1.0:
        print(f"  [OK] Sharpe ratio > 1.0: {sharpe_ratio:.2f}")
    elif sharpe_ratio > 0:
        print(f"  [WARN] Sharpe ratio <= 1.0: {sharpe_ratio:.2f}")
    else:
        print(f"  [CRITICAL] Negative Sharpe ratio: {sharpe_ratio:.2f}")
    
    if max_max_dd < 0.1:
        print(f"  [OK] Max drawdown < 10%: {max_max_dd*100:.2f}%")
    elif max_max_dd < 0.2:
        print(f"  [WARN] Max drawdown >= 10%: {max_max_dd*100:.2f}%")
    else:
        print(f"  [CRITICAL] Max drawdown >= 20%: {max_max_dd*100:.2f}%")
    
    print("\n" + "="*80)
    
    return {
        "mean_reward": mean_reward,
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "total_return": total_return,
        "episode_win_rate": episode_win_rate,
        "mean_win_rate": mean_win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_max_dd,
        "total_trades": total_trades
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Model path")
    parser.add_argument("--config", type=str, default="configs/train_config_adaptive.yaml", help="Config path")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.config, args.episodes)

