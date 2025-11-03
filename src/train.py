"""
Training Script for RL Trading Agent

This script trains the PPO agent on historical trading data.

Usage:
    python src/train.py --config configs/train_config.yaml
    python src/train.py --config configs/train_config.yaml --device cuda
"""

import argparse
import yaml
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Handles the training loop"""
    
    def __init__(self, config: dict, checkpoint_path: str = None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Validate and adjust device selection
        requested_device = config["training"]["device"]
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA not available (PyTorch not compiled with CUDA support). Using CPU instead.")
                self.device = "cpu"
                config["training"]["device"] = "cpu"  # Update config for consistency
            else:
                # CUDA is available - verify it works and get device info
                try:
                    # Try to create a test tensor on CUDA to verify it works
                    test_tensor = torch.tensor([1.0]).cuda()
                    gpu_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda
                    print(f"✅ Using GPU: {gpu_name} (CUDA {cuda_version})")
                    self.device = "cuda"
                except Exception as e:
                    print(f"⚠️  CUDA device error: {e}. Using CPU instead.")
                    self.device = "cpu"
                    config["training"]["device"] = "cpu"
        else:
            self.device = requested_device
        
        print(f"Training device: {self.device}")
        
        # Setup paths
        self.log_dir = Path(config["logging"]["log_dir"])
        self.model_dir = Path("models")
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        if config["logging"]["tensorboard"]:
            run_name = f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(self.log_dir / run_name))
        else:
            self.writer = None
        
        # Load data
        print("Loading data...")
        self._load_data()
        
        # Create environment
        print("Creating trading environment...")
        # Get max_episode_steps from config (default 10000 to ensure episodes complete in reasonable time)
        max_episode_steps = config["environment"].get("max_episode_steps", 10000)
        print(f"  Max episode steps: {max_episode_steps} (episodes will terminate at this limit)")
        
        self.env = TradingEnvironment(
            data=self.multi_tf_data,
            timeframes=config["environment"]["timeframes"],
            initial_capital=config["risk_management"]["initial_capital"],
            transaction_cost=config["risk_management"]["commission"] / config["risk_management"]["initial_capital"],
            reward_config=config["environment"]["reward"],
            max_episode_steps=max_episode_steps  # Limit episode length for reasonable training
        )
        
        # Create agent
        print("Creating PPO agent...")
        model_config = config["model"]
        
        # Get network architecture (if specified, else use default)
        hidden_dims = model_config.get("hidden_dims", [256, 256, 128])
        
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_range=tuple(config["environment"]["action_range"]),
            learning_rate=model_config["learning_rate"],
            gamma=model_config["gamma"],
            gae_lambda=model_config["gae_lambda"],
            clip_range=model_config["clip_range"],
            value_loss_coef=model_config["value_loss_coef"],
            entropy_coef=model_config["entropy_coef"],
            device=self.device,
            hidden_dims=hidden_dims
        )
        
        # Mixed precision training (FP16) for 2x speedup on modern GPUs
        self.use_mixed_precision = config["training"].get("use_mixed_precision", False)
        if self.use_mixed_precision and self.device == "cuda":
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler()
            self.autocast = autocast
            print("✅ Mixed precision (FP16) enabled - expect ~2x speedup")
        else:
            self.scaler = None
            self.autocast = None
        
        # Training parameters
        self.total_timesteps = config["training"]["total_timesteps"]
        self.save_freq = config["training"]["save_freq"]
        self.eval_freq = config["training"]["eval_freq"]
        
        # Metrics
        self.timestep = 0
        self.episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_metrics = {}
        
        # Current episode tracking (for in-progress episodes)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Load performance mode from settings (for dynamic adjustment during training)
        self.performance_mode = self._load_performance_mode()
        print(f"⚙️  Performance mode: {self.performance_mode}")
        
        # Load checkpoint if provided (resume training)
        if self.checkpoint_path:
            # Normalize path - handle Windows backslashes properly
            # First, normalize backslashes to forward slashes
            normalized = str(self.checkpoint_path).replace('\\', '/')
            checkpoint_path = Path(normalized)
            
            # If path doesn't exist, try making it relative to project root
            if not checkpoint_path.exists():
                # Get project root (assuming we're in src/, go up one level)
                project_root = Path(__file__).parent.parent
                checkpoint_path = project_root / normalized.lstrip('/')
            
            # Final check
            if checkpoint_path.exists():
                print(f"📂 Resuming from checkpoint: {checkpoint_path}")
                print(f"   Absolute path: {checkpoint_path.resolve()}")
                timestep, episode, rewards, lengths = self.agent.load_with_training_state(str(checkpoint_path))
                self.timestep = timestep
                self.episode = episode
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                print(f"✅ Resume: timestep={timestep}, episode={episode}, rewards={len(rewards)}")
            else:
                print(f"⚠️  WARNING: Checkpoint path does not exist!")
                print(f"   Original path: {self.checkpoint_path}")
                print(f"   Normalized path: {normalized}")
                print(f"   Attempted absolute: {checkpoint_path.resolve()}")
                print(f"   Current working directory: {Path.cwd()}")
                print(f"   Project root: {Path(__file__).parent.parent}")
                print(f"   Starting fresh training instead.")
        
    def _load_performance_mode(self):
        """Load performance mode from settings.json"""
        settings_file = Path("settings.json")
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get("performance_mode", "quiet")
            except:
                pass
        return "quiet"  # Default: resource-friendly mode
    
    def _load_data(self):
        """Load multi-timeframe data"""
        # Get NT8 data path from config or settings
        nt8_data_path = self.config.get("data", {}).get("nt8_data_path")
        if not nt8_data_path:
            # Try to load from settings.json
            settings_file = Path("settings.json")
            if settings_file.exists():
                try:
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        nt8_data_path = settings.get("nt8_data_path")
                except:
                    pass
        
        extractor = DataExtractor(nt8_data_path=nt8_data_path)
        instrument = self.config["environment"]["instrument"]
        timeframes = self.config["environment"]["timeframes"]
        
        # Try to load data
        try:
            self.multi_tf_data = extractor.load_multi_timeframe_data(
                instrument, timeframes
            )
            print(f"Loaded data for {instrument} with timeframes: {timeframes}")
            
            # Print data stats
            for tf, df in self.multi_tf_data.items():
                print(f"  {tf}min: {len(df)} bars, "
                      f"from {df['timestamp'].min()} to {df['timestamp'].max()}")
        except FileNotFoundError:
            print("\n⚠️  Data files not found!")
            print(f"Please export data from NT8 and save as:")
            for tf in timeframes:
                print(f"  data/raw/{instrument}_{tf}min.csv")
            raise
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Timeframes: {self.config['environment']['timeframes']}")
        print(f"Instrument: {self.config['environment']['instrument']}")
        print("="*60 + "\n")
        
        # Reset environment
        state, info = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        best_mean_reward = float('-inf')
        
        # Progress bar
        pbar = tqdm(total=self.total_timesteps, desc="Training")
        
        while self.timestep < self.total_timesteps:
            # Select action
            action, value, log_prob = self.agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            # Debug: Log environment state vs training episode length
            if episode_length >= 9995:
                import sys
                env_step = getattr(self.env, 'current_step', 'unknown')
                env_max = getattr(self.env, 'max_steps', 'unknown')
                print(f"[DEBUG] Step comparison: episode_length={episode_length}, env.current_step={env_step}, env.max_steps={env_max}, terminated={terminated}", flush=True)
                sys.stdout.flush()
            
            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            self.current_episode_reward = episode_reward
            self.current_episode_length = episode_length
            self.timestep += 1
            pbar.update(1)
            
            # Debug: Log reward accumulation at start of new episode
            if episode_length <= 10 or episode_length % 1000 == 0:
                import sys
                print(f"[DEBUG] Episode {self.episode + 1}: step={episode_length}, cumulative_reward={episode_reward:.4f}, step_reward={reward:.4f}", flush=True)
                sys.stdout.flush()
            
            # Debug: Log if episode is about to terminate
            if done or episode_length >= 9995:
                import sys
                print(f"\n[DEBUG] Train: episode_length={episode_length}, done={done}, terminated={terminated}, truncated={truncated}, cumulative_reward={episode_reward:.2f}, step_reward={reward:.4f}", flush=True)
                sys.stdout.flush()
            
            # Update agent if buffer is full or episode ended
            if done or len(self.agent.states) >= self.config["model"]["n_steps"]:
                if len(self.agent.states) > 0:
                    # Reload performance mode for dynamic adjustment
                    self.performance_mode = self._load_performance_mode()
                    
                    # Calculate dynamic batch size and epochs based on performance mode
                    base_batch_size = self.config["model"]["batch_size"]
                    base_n_epochs = self.config["model"].get("n_epochs", 10)
                    
                    if self.performance_mode == "performance":
                        # Use larger batch size and more epochs for faster training
                        dynamic_batch_size = base_batch_size * 2
                        dynamic_n_epochs = int(base_n_epochs * 1.5)
                    else:  # "quiet" mode - default
                        # Use configured batch size and epochs
                        dynamic_batch_size = base_batch_size
                        dynamic_n_epochs = base_n_epochs
                    
                    # Update agent
                    metrics = self.agent.update(
                        n_epochs=dynamic_n_epochs,
                        batch_size=dynamic_batch_size,
                        scaler=self.scaler,
                        autocast=self.autocast
                    )
                    
                    # Store metrics for API access (convert numpy/torch types to native Python types)
                    self.last_update_metrics = {
                        "loss": float(metrics.get("loss", 0.0)),
                        "policy_loss": float(metrics.get("policy_loss", 0.0)),
                        "value_loss": float(metrics.get("value_loss", 0.0)),
                        "entropy": float(metrics.get("entropy", 0.0)),
                    }
                    
                    # Log metrics
                    if self.writer:
                        for key, value in metrics.items():
                            self.writer.add_scalar(f"train/{key}", value, self.timestep)
            
            # Handle episode end
            if done:
                # Debug: Log episode completion
                import sys
                print(f"\n[DEBUG] Episode completing: length={episode_length}, reward={episode_reward:.2f}, terminated={terminated}, truncated={truncated}", flush=True)
                sys.stdout.flush()
                
                self.episode += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Log episode metrics
                if self.writer:
                    self.writer.add_scalar("episode/reward", episode_reward, self.episode)
                    self.writer.add_scalar("episode/length", episode_length, self.episode)
                    self.writer.add_scalar("episode/trades", step_info.get("trades", 0), self.episode)
                    self.writer.add_scalar("episode/pnl", step_info.get("pnl", 0), self.episode)
                    self.writer.add_scalar("episode/equity", step_info.get("equity", 0), self.episode)
                
                # Print episode summary
                if self.episode % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    
                    print(f"\nEpisode {self.episode} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Length: {episode_length} | "
                          f"PnL: ${step_info.get('pnl', 0):.2f} | "
                          f"Trades: {step_info.get('trades', 0)}")
                    print(f"  Last 10 episodes - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")
                    
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        print(f"  🎉 New best mean reward: {best_mean_reward:.2f}")
                
                # Reset for next episode
                state, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
            else:
                state = next_state
            
            # Save checkpoint
            if self.timestep % self.save_freq == 0:
                checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
                self.agent.save_with_training_state(
                    str(checkpoint_path),
                    self.timestep,
                    self.episode,
                    self.episode_rewards,
                    self.episode_lengths
                )
                
                # Save best model
                if len(self.episode_rewards) > 0:
                    recent_mean = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                    if recent_mean > best_mean_reward:
                        best_path = self.model_dir / "best_model.pt"
                        self.agent.save_with_training_state(
                            str(best_path),
                            self.timestep,
                            self.episode,
                            self.episode_rewards,
                            self.episode_lengths
                        )
            
            # Evaluation
            if self.timestep % self.eval_freq == 0 and self.timestep > 0:
                self._evaluate()
        
        pbar.close()
        
        # Final save
        final_path = self.model_dir / "final_model.pt"
        self.agent.save_with_training_state(
            str(final_path),
            self.timestep,
            self.episode,
            self.episode_rewards,
            self.episode_lengths
        )
        
        # Save training summary
        self._save_training_summary()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total episodes: {self.episode}")
        print(f"Mean reward: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}")
        print(f"Best model saved to: models/best_model.pt")
        if self.writer:
            print(f"TensorBoard logs: {self.writer.log_dir}")
        print("="*60)
    
    def _evaluate(self):
        """Evaluate agent performance"""
        # Use same max_episode_steps for evaluation
        max_episode_steps = self.config["environment"].get("max_episode_steps", 10000)
        eval_env = TradingEnvironment(
            data=self.multi_tf_data,
            timeframes=self.config["environment"]["timeframes"],
            initial_capital=self.config["risk_management"]["initial_capital"],
            transaction_cost=self.config["risk_management"]["commission"] / self.config["risk_management"]["initial_capital"],
            reward_config=self.config["environment"]["reward"],
            max_episode_steps=max_episode_steps
        )
        
        eval_rewards = []
        eval_pnls = []
        
        for _ in range(5):  # Run 5 evaluation episodes
            state, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_pnls.append(info.get("pnl", 0))
        
        mean_reward = np.mean(eval_rewards)
        mean_pnl = np.mean(eval_pnls)
        
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", mean_reward, self.timestep)
            self.writer.add_scalar("eval/mean_pnl", mean_pnl, self.timestep)
        
        print(f"\n📊 Evaluation @ step {self.timestep}: "
              f"Mean reward: {mean_reward:.2f}, Mean PnL: ${mean_pnl:.2f}")
    
    def _save_training_summary(self):
        """Save training summary to JSON"""
        summary = {
            "total_timesteps": self.timestep,
            "total_episodes": self.episode,
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "std_reward": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
            "best_reward": float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
        }
        
        summary_path = self.log_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda). Overrides config."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config["training"]["device"] = args.device
    
    # Check device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        config["training"]["device"] = "cpu"
    
    # Create trainer (with checkpoint if specified for resume)
    trainer = Trainer(config, checkpoint_path=args.checkpoint)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

