"""
Training Script for RL Trading Agent

This script trains the PPO agent on historical trading data.

Usage:
    python src/train.py --config configs/train_config_full.yaml
    python src/train.py --config configs/train_config_full.yaml --device cuda
"""

import sys
from pathlib import Path

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import yaml
import os
import numpy as np
import torch
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import re

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter
from src.trading_hours import TradingHoursManager
from src.utils.colors import error, warn, success, info


class Trainer:
    """Handles the training loop"""
    
    def __init__(self, config: dict, checkpoint_path: str = None, config_path: str = None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path  # Store config_path for adaptive trainer
        
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
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✅ Using GPU: {gpu_name} (CUDA {cuda_version})")
                    print(f"   GPU Memory: {gpu_memory:.1f} GB total")
                    print(f"   Device will be used for: Actor/Critic networks, tensor operations")
                    self.device = "cuda"
                except Exception as e:
                    print(error(f"⚠️  CUDA device error: {e}. Using CPU instead."))
                    self.device = "cpu"
                    config["training"]["device"] = "cpu"
        else:
            self.device = requested_device
        
        print(f"Training device: {self.device}")
        
        # Setup paths
        self.log_dir = Path(config["logging"]["log_dir"])
        self.model_dir = Path("models")
        self.archive_dir = Path("models/Archive")
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Archive existing models if starting fresh training (no checkpoint provided)
        # This prevents accidentally reusing old models
        should_archive = (
            not checkpoint_path and  # No checkpoint provided (starting fresh)
            not config["training"].get("transfer_learning", False)  # Not using transfer learning
        )
        if should_archive:
            self._archive_existing_models()
        
        # TensorBoard writer
        if config["logging"]["tensorboard"]:
            run_name = f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(self.log_dir / run_name))
        else:
            self.writer = None
        
        # Load data
        import time
        data_load_start = time.time()
        print("Loading data...")
        print(f"  Instrument: {config['environment']['instrument']}")
        print(f"  Timeframes: {config['environment']['timeframes']}")
        print("  This may take a while if loading many files...")
        sys.stdout.flush()  # Ensure output is visible immediately
        
        try:
            # Add periodic progress updates during data loading
            import threading
            progress_thread = None
            progress_stop = threading.Event()
            
            def progress_reporter():
                """Report progress every 10 seconds during data loading"""
                elapsed = 0
                while not progress_stop.is_set():
                    time.sleep(10)
                    if not progress_stop.is_set():
                        elapsed = time.time() - data_load_start
                        print(f"  [Progress] Still loading data... ({elapsed:.0f}s elapsed)")
                        sys.stdout.flush()
            
            progress_thread = threading.Thread(target=progress_reporter, daemon=True)
            progress_thread.start()
            
            try:
                self._load_data()
                progress_stop.set()
                data_load_elapsed = time.time() - data_load_start
                print(f"[OK] Data loaded successfully (took {data_load_elapsed:.1f}s)")
                sys.stdout.flush()
            finally:
                progress_stop.set()
                if progress_thread and progress_thread.is_alive():
                    progress_thread.join(timeout=1)
        except Exception as e:
            progress_stop.set() if 'progress_stop' in locals() else None
            data_load_elapsed = time.time() - data_load_start
            print(error(f"[ERROR] Error loading data after {data_load_elapsed:.1f}s: {e}"))
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        # Create environment
        print("Creating trading environment...")
        sys.stdout.flush()  # Force flush before creating environment
        # Get max_episode_steps from config (default 10000 to ensure episodes complete in reasonable time)
        max_episode_steps = config["environment"].get("max_episode_steps", 10000)
        print(f"  Max episode steps: {max_episode_steps} (episodes will terminate at this limit)")
        sys.stdout.flush()
        
        self.env = TradingEnvironment(
            data=self.multi_tf_data,
            timeframes=config["environment"]["timeframes"],
            initial_capital=config["risk_management"]["initial_capital"],
            transaction_cost=config["risk_management"]["commission"] / config["risk_management"]["initial_capital"],
            reward_config=config["environment"]["reward"],
            max_episode_steps=max_episode_steps,  # Limit episode length for reasonable training
            action_threshold=config["environment"].get("action_threshold", 0.05)  # Configurable action threshold (default 5%)
        )
        
        # Setup trading journal integration (non-intrusive)
        try:
            from src.journal_integration import get_integration
            journal_integration = get_integration()
            journal_integration.start(self)
            journal_integration.setup_trade_callback(self.env)
            self.journal_integration = journal_integration
            print("[OK] Trading Journal integration enabled")
        except Exception as e:
            print(warn(f"[WARN] Trading Journal integration failed: {e}"))
            self.journal_integration = None
        
        # Force flush after environment creation to ensure Priority 1 messages appear
        sys.stdout.flush()
        
        # Create agent
        print("Creating PPO agent...")
        model_config = config["model"]
        
        # Check if we're resuming from checkpoint - if so, load architecture from checkpoint
        checkpoint_hidden_dims = None
        checkpoint_state_dim = None
        if self.checkpoint_path:
            checkpoint_path_obj = Path(str(self.checkpoint_path).replace('\\', '/'))
            if not checkpoint_path_obj.exists():
                project_root = Path(__file__).parent.parent
                checkpoint_path_obj = project_root / str(self.checkpoint_path).replace('\\', '/').lstrip('/')
            
            if checkpoint_path_obj.exists():
                try:
                    # torch is already imported at the top of the file
                    checkpoint = torch.load(str(checkpoint_path_obj), map_location='cpu', weights_only=False)
                    checkpoint_hidden_dims = checkpoint.get("hidden_dims")
                    checkpoint_state_dim = checkpoint.get("state_dim")
                    
                    if checkpoint_hidden_dims:
                        print(f"📐 Found architecture in checkpoint: hidden_dims={checkpoint_hidden_dims}")
                    else:
                        # Try to infer from state_dict shapes (for old checkpoints)
                        print(f"⚠️  Checkpoint doesn't have hidden_dims saved, trying to infer from model weights...")
                        actor_state = checkpoint.get("actor_state_dict", {})
                        if actor_state:
                            # Infer hidden_dims from feature_layers shapes
                            inferred_dims = []
                            # feature_layers.0.weight should be [hidden_dim, state_dim]
                            if "feature_layers.0.weight" in actor_state:
                                first_layer_shape = actor_state["feature_layers.0.weight"].shape
                                inferred_dims.append(first_layer_shape[0])
                                
                                # Check subsequent layers
                                layer_idx = 3  # Skip ReLU and Dropout
                                while f"feature_layers.{layer_idx}.weight" in actor_state:
                                    layer_shape = actor_state[f"feature_layers.{layer_idx}.weight"].shape
                                    inferred_dims.append(layer_shape[0])
                                    layer_idx += 3  # Skip ReLU and Dropout between Linear layers
                                
                                if inferred_dims:
                                    checkpoint_hidden_dims = inferred_dims
                                    print(f"✅ Inferred architecture from checkpoint: hidden_dims={checkpoint_hidden_dims}")
                                    checkpoint_state_dim = checkpoint.get("state_dim")
                                else:
                                    print(f"⚠️  Could not infer architecture, will use config/default")
                            else:
                                print(f"⚠️  Could not find feature layers in checkpoint")
                except Exception as e:
                    print(f"⚠️  Could not read checkpoint architecture: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"   Will use config/default architecture")

        # Determine architecture to use for agent initialization
        config_hidden_dims = model_config.get("hidden_dims", [256, 256, 128])
        current_state_dim = self.env.state_dim
        architecture_matches_config = False
        if checkpoint_hidden_dims and checkpoint_state_dim is not None:
            architecture_matches_config = (
                checkpoint_hidden_dims == config_hidden_dims and
                checkpoint_state_dim == current_state_dim
            )
        elif checkpoint_hidden_dims:
            architecture_matches_config = checkpoint_hidden_dims == config_hidden_dims
        
        if self.checkpoint_path and checkpoint_hidden_dims and not architecture_matches_config:
            print(f"⚠️  Checkpoint architecture differs from config. Initializing agent with config architecture: {config_hidden_dims}")
            hidden_dims = config_hidden_dims
        elif checkpoint_hidden_dims:
            hidden_dims = checkpoint_hidden_dims
            print(f"   Using architecture from checkpoint: {hidden_dims}")
        else:
            hidden_dims = config_hidden_dims
            print(f"   Using architecture from config/default: {hidden_dims}")
        
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
        
        # Supervised pre-training (if enabled)
        pretrain_config = config.get("pretraining", {})
        self.pretrained = False
        if pretrain_config.get("enabled", False) and not self.checkpoint_path:
            # Only pre-train if starting fresh (no checkpoint resume)
            try:
                from src.supervised_pretraining import SupervisedPretrainer
                print(info("\n" + "="*70))
                print(info("SUPERVISED PRE-TRAINING"))
                print(info("="*70))
                print(info("Pre-training actor network on historical data before RL fine-tuning..."))
                print(info("This helps the agent learn basic market patterns and reduces random exploration."))
                
                pretrainer = SupervisedPretrainer(
                    config=config,
                    data_extractor=self.data_extractor,
                    device=self.device
                )
                pretrain_metrics = pretrainer.run_pretraining(
                    actor=self.agent.actor,
                    env=self.env
                )
                if pretrain_metrics:
                    self.pretrained = True
                    best_val_loss = pretrain_metrics.get('best_val_loss', 'N/A')
                    epochs_trained = pretrain_metrics.get('epochs_trained', 'N/A')
                    print(success(f"\n[PRETRAIN] Pre-training completed successfully!"))
                    print(info(f"  Best validation loss: {best_val_loss:.6f}"))
                    print(info(f"  Epochs trained: {epochs_trained}"))
                    print(info("  Pre-trained weights are now in the actor network."))
                    print(info("  RL training will fine-tune these weights."))
                    print(info("="*70 + "\n"))
                    
                    # Optionally save pre-trained weights separately for reference
                    if pretrain_config.get("save_pretrained_weights", False):
                        pretrained_path = self.model_dir / "pretrained_actor.pt"
                        torch.save({
                            "actor_state_dict": self.agent.actor.state_dict(),
                            "pretrain_metrics": pretrain_metrics,
                            "config": pretrain_config
                        }, pretrained_path)
                        print(success(f"[PRETRAIN] Pre-trained weights saved to: {pretrained_path}"))
                else:
                    print(warn("[PRETRAIN] Pre-training returned no metrics (may be disabled in config)"))
            except Exception as e:
                print(error(f"[PRETRAIN] Pre-training failed: {e}"))
                print(warn("[PRETRAIN] Continuing with RL training from random initialization..."))
                import traceback
                traceback.print_exc()
        elif pretrain_config.get("enabled", False) and self.checkpoint_path:
            print(info("[PRETRAIN] Skipping pre-training (resuming from checkpoint)"))
        elif not pretrain_config.get("enabled", False):
            print(info("[PRETRAIN] Pre-training disabled in config (pretraining.enabled: false)"))
        
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
        
        # Early stopping configuration
        early_stopping_config = config["training"].get("early_stopping", {})
        self.early_stopping_enabled = early_stopping_config.get("enabled", False)
        self.early_stopping_patience = early_stopping_config.get("patience", 50000)  # Steps without improvement
        self.early_stopping_min_delta = early_stopping_config.get("min_delta", 0.005)  # Minimum improvement threshold
        self.early_stopping_best_metric = float('-inf')  # Track best performance metric
        self.early_stopping_last_improvement = 0  # Timestep of last improvement
        self.early_stopping_metric_name = "mean_reward"  # Metric to track (can be "mean_reward" or "win_rate")
        
        if self.early_stopping_enabled:
            print(f"✅ Early stopping enabled:")
            print(f"   Patience: {self.early_stopping_patience:,} timesteps")
            print(f"   Min delta: {self.early_stopping_min_delta:.4f}")
            print(f"   Metric: {self.early_stopping_metric_name}")
        
        # Metrics
        self.timestep = 0
        self.episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_metrics = {}
        
        # Current episode tracking (for in-progress episodes)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Trading metrics tracking
        self.episode_trades = []  # List of trade counts per episode
        self.episode_pnls = []  # List of PnL per episode
        self.episode_equities = []  # List of final equity per episode
        self.episode_win_rates = []  # List of win rates per episode
        self.episode_max_drawdowns = []  # List of max drawdowns per episode
        self.total_trades = 0  # Cumulative total trades across all episodes
        self.total_winning_trades = 0  # Cumulative winning trades
        self.total_losing_trades = 0  # Cumulative losing trades
        self.current_episode_trades = 0  # Current episode trade count
        self.current_episode_pnl = 0.0  # Current episode PnL
        self.current_episode_equity = 0.0  # Current episode equity
        self.current_episode_win_rate = 0.0  # Current episode win rate
        self.current_episode_max_drawdown = 0.0  # Current episode max drawdown
        
        # Track average win/loss and risk/reward ratio for profitability monitoring
        self.current_avg_win = 0.0  # Current average winning trade PnL
        self.current_avg_loss = 0.0  # Current average losing trade PnL
        self.current_risk_reward_ratio = 0.0  # Current risk/reward ratio
        
        # DecisionGate integration for training (ensures consistency with live trading)
        self.decision_gate = None
        self.decision_gate_enabled = config.get("training", {}).get("use_decision_gate", False)
        if self.decision_gate_enabled:
            from src.decision_gate import DecisionGate
            decision_gate_config = config.get("decision_gate", {})
            # For training, ALWAYS set min_confluence_required to 0 to allow RL-only trades
            # During training, we don't use swarm, so confluence_count will always be 0
            # Quality filters (confidence, quality score, expected value) will still be applied
            training_decision_gate_config = decision_gate_config.copy()
            training_decision_gate_config["min_confluence_required"] = 0  # Always allow RL-only trades during training
            training_decision_gate_config["swarm_enabled"] = False  # Disable swarm during training (not used anyway)
            
            # Set min_combined_confidence for quality filtering during training
            # Increased from 0.3 to 0.5 to filter for quality trades
            if training_decision_gate_config.get("min_combined_confidence", 0.7) >= 0.5:
                training_decision_gate_config["min_combined_confidence"] = 0.5  # Quality threshold - requires 50% confidence minimum
                print("   [DecisionGate] Set min_combined_confidence to 0.5 for quality filtering")
            
            print("   [DecisionGate] Training mode: min_confluence_required=0, swarm_enabled=false")
            print("   [DecisionGate] Quality filters (confidence, quality score, EV) still applied")
            
            self.decision_gate = DecisionGate(training_decision_gate_config)
            print("[OK] DecisionGate integrated into training loop")
            print(f"   Will apply quality filters, expected value checks, and position sizing")
            print(f"   Ensures consistency between training and live trading")
        
        # Adaptive training system
        self.adaptive_trainer = None
        if config.get("training", {}).get("adaptive_training", {}).get("enabled", False):
            from src.adaptive_trainer import AdaptiveTrainer, AdaptiveConfig
            adaptive_cfg = AdaptiveConfig(
                eval_frequency=config["training"]["adaptive_training"].get("eval_frequency", 5000),
                eval_episodes=config["training"]["adaptive_training"].get("eval_episodes", 3),
                min_trades_per_episode=config["training"]["adaptive_training"].get("min_trades_per_episode", 0.5),
                auto_save_on_improvement=config["training"]["adaptive_training"].get("auto_save_on_improvement", True),
                improvement_threshold=config["training"]["adaptive_training"].get("improvement_threshold", 0.05)
            )
            self.adaptive_trainer = AdaptiveTrainer(str(self.config_path), adaptive_cfg)
            print("✅ Adaptive training system enabled")
            print(f"   Will evaluate every {adaptive_cfg.eval_frequency:,} timesteps")
            print(f"   Auto-adjusts: entropy_coef, inaction_penalty, learning_rate")
        
        # Load performance mode from settings (for dynamic adjustment during training)
        self.performance_mode = self._load_performance_mode()
        
        # Adaptive Turbo mode: Track multipliers for dynamic adjustment
        # Target: 65% GPU utilization, <8GB VRAM
        # Start at MAXIMUM since VRAM is only 11-14% - model is severely underutilized
        self.turbo_batch_multiplier = 50.0  # Start at MAXIMUM (50x = 6400 batch size for base 128)
        self.turbo_epoch_multiplier = 8.0   # Start VERY aggressive (8x = 240 epochs with base 30)
        self.gpu_target_util = 65.0  # Target GPU utilization %
        self.vram_limit_gb = 8.0    # VRAM limit in GB
        self.max_batch_multiplier = 100.0  # Increased max multiplier (for very small models with lots of VRAM)
        
        if self.performance_mode == "turbo":
            print(f"[PERF] Performance mode: {self.performance_mode} (ADAPTIVE GPU UTILIZATION)")
            print(f"   Adaptive Turbo: Targeting {self.gpu_target_util}% GPU, <{self.vram_limit_gb}GB VRAM")
            print(f"   Starting: {self.turbo_batch_multiplier}x batch, {self.turbo_epoch_multiplier}x epochs")
        else:
            print(f"[PERF] Performance mode: {self.performance_mode}")
        
        # Load checkpoint if provided (resume training)
        checkpoint_timestep = 0
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
                
                # Check for architecture mismatch
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                checkpoint_hidden_dims = checkpoint.get("hidden_dims")
                checkpoint_state_dim = checkpoint.get("state_dim")
                
                # Get current architecture
                current_hidden_dims = model_config.get("hidden_dims", [256, 256, 128])
                current_state_dim = self.env.state_dim
                
                # Check if architectures match
                architecture_matches = (
                    checkpoint_hidden_dims == current_hidden_dims and
                    checkpoint_state_dim == current_state_dim
                )
                
                if not architecture_matches:
                    print(f"\n⚠️  Architecture mismatch detected!")
                    print(f"   Checkpoint: state_dim={checkpoint_state_dim}, hidden_dims={checkpoint_hidden_dims}")
                    print(f"   Current:    state_dim={current_state_dim}, hidden_dims={current_hidden_dims}")
                    print(f"   🔄 Using transfer learning to preserve learned knowledge...")
                    
                    # Use transfer learning
                    transfer_strategy = config.get("training", {}).get("transfer_strategy", "copy_and_extend")
                    result = self.agent.load_with_transfer(
                        str(checkpoint_path),
                        transfer_strategy=transfer_strategy
                    )
                    # load_with_transfer returns tuple: (timestep, episode, rewards, lengths)
                    timestep, episode, rewards, lengths = result[:4]
                    # For transfer learning, initialize metrics as empty (fresh start for new architecture)
                    # Try to load from checkpoint if available
                    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                    pnls = checkpoint.get("episode_pnls", [])
                    equities = checkpoint.get("episode_equities", [])
                    win_rates = checkpoint.get("episode_win_rates", [])
                else:
                    # Architectures match, use normal loading
                    timestep, episode, rewards, lengths, pnls, equities, win_rates = self.agent.load_with_training_state(str(checkpoint_path))
                
                checkpoint_timestep = timestep
                self.timestep = timestep
                self.episode = episode
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                self.episode_pnls = pnls  # Initialize from checkpoint
                self.episode_equities = equities  # Initialize from checkpoint
                self.episode_win_rates = win_rates  # Initialize from checkpoint
                print(f"✅ Resume: timestep={timestep}, episode={episode}, rewards={len(rewards)}, pnls={len(pnls)}, equities={len(equities)}, win_rates={len(win_rates)}")
                
                # If we've already reached or exceeded total_timesteps, continue training
                # by adding additional timesteps equal to the original goal
                if self.timestep >= self.total_timesteps:
                    additional_timesteps = config["training"]["total_timesteps"]
                    self.total_timesteps = self.timestep + additional_timesteps
                    print(f"📊 Checkpoint already at {checkpoint_timestep:,} timesteps (goal: {config['training']['total_timesteps']:,})")
                    print(f"   Continuing training for {additional_timesteps:,} more timesteps")
                    print(f"   New goal: {self.total_timesteps:,} timesteps")
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
                    turbo_enabled = settings.get("turbo_training_mode", False)
                    perf_mode = settings.get("performance_mode", "quiet")
                    
                    # Check for turbo mode first (overrides performance mode)
                    if turbo_enabled:
                        print(f"[OK] TURBO MODE DETECTED in settings.json (turbo_training_mode: {turbo_enabled})")
                        return "turbo"
                    else:
                        print(f"[PERF] Performance mode: {perf_mode} (turbo_training_mode: {turbo_enabled})")
                        return perf_mode
            except Exception as e:
                print(warn(f"[WARN] Error loading performance mode: {e}"))
                pass
        print(f"⚠️  settings.json not found, defaulting to quiet mode")
        return "quiet"  # Default: resource-friendly mode
    
    def _get_gpu_stats(self):
        """Get GPU utilization and VRAM usage using nvidia-smi"""
        if self.device != "cuda":
            return None
        
        try:
            # Query GPU utilization and memory
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Parse output: "utilization.gpu, memory.used, memory.total"
                match = re.match(r'(\d+),\s*(\d+),\s*(\d+)', result.stdout.strip())
                if match:
                    gpu_util = float(match.group(1))
                    vram_used_mb = float(match.group(2))
                    vram_total_mb = float(match.group(3))
                    vram_used_gb = vram_used_mb / 1024.0
                    return {
                        'utilization': gpu_util,
                        'vram_used_gb': vram_used_gb,
                        'vram_total_gb': vram_total_mb / 1024.0
                    }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception) as e:
            # Fallback to PyTorch memory only
            try:
                vram_used_gb = torch.cuda.memory_allocated(0) / 1e9
                return {
                    'utilization': None,  # Can't get from PyTorch
                    'vram_used_gb': vram_used_gb,
                    'vram_total_gb': None
                }
            except:
                pass
        
        return None
    
    def _adjust_turbo_multipliers(self, gpu_stats_before, gpu_stats_after):
        """Adjust Turbo mode multipliers based on GPU usage"""
        if self.performance_mode != "turbo" or not gpu_stats_after:
            return
        
        gpu_util = gpu_stats_after.get('utilization')
        vram_used = gpu_stats_after.get('vram_used_gb')
        
        # Calculate gap from target (for use in adjustment logic)
        gap_from_target = self.gpu_target_util - gpu_util if gpu_util is not None else 50  # Assume 50% gap if unknown
        
        # Dynamic adjustment rate based on how far from target AND VRAM headroom
        # If GPU is very low AND VRAM is low, make very aggressive adjustments
        vram_percent = (vram_used / self.vram_limit_gb * 100) if (vram_used is not None and self.vram_limit_gb > 0) else 100
        
        if gpu_util is not None:
            if gap_from_target > 50:  # GPU < 15% (very far from 65%)
                if vram_percent < 20:  # VRAM < 20% (<1.6GB) - lots of headroom
                    adjustment_rate = 1.0  # 100% increase - very aggressive!
                elif vram_percent < 40:  # VRAM < 40% (<3.2GB)
                    adjustment_rate = 0.7  # 70% increase
                else:
                    adjustment_rate = 0.5  # 50% increase
            elif gap_from_target > 30:  # GPU < 35%
                if vram_percent < 30:
                    adjustment_rate = 0.5  # 50% increase
                else:
                    adjustment_rate = 0.3  # 30% increase
            elif gap_from_target > 15:  # GPU < 50%
                adjustment_rate = 0.2  # 20% increase
            else:
                adjustment_rate = 0.1  # 10% increase (normal)
        else:
            # Can't read GPU, but if VRAM is low, be aggressive
            adjustment_rate = 0.5 if vram_percent < 30 else 0.2
        
        # Adjust based on GPU utilization
        if gpu_util is not None:
            if gpu_util > self.gpu_target_util + 5:  # Too high (>70%)
                # Reduce batch size (more impact on GPU)
                self.turbo_batch_multiplier = max(2.0, self.turbo_batch_multiplier * (1 - 0.1))
                print(f"   [DOWN] GPU too high ({gpu_util:.1f}% > {self.gpu_target_util}%), reducing batch multiplier to {self.turbo_batch_multiplier:.2f}x")
            elif gpu_util < self.gpu_target_util:  # Too low (<65%)
                # Increase batch size - more aggressive when far from target
                old_batch = self.turbo_batch_multiplier
                old_epoch = self.turbo_epoch_multiplier
                
                # Use dynamic max multiplier (up to 50x for very small models with lots of VRAM)
                self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * (1 + adjustment_rate))
                
                # Also increase epochs if GPU is very low (more work per update)
                if gap_from_target > 30:  # GPU < 35%
                    epoch_adjustment = 0.2  # 20% increase in epochs
                    self.turbo_epoch_multiplier = min(5.0, self.turbo_epoch_multiplier * (1 + epoch_adjustment))
                
                if self.turbo_batch_multiplier != old_batch or self.turbo_epoch_multiplier != old_epoch:
                    changes = []
                    if self.turbo_batch_multiplier != old_batch:
                        vram_free = 100 - vram_percent if vram_used is not None else 0
                        vram_info = f" (VRAM {vram_free:.1f}% free)" if vram_used is not None else ""
                        changes.append(f"batch {old_batch:.2f}x → {self.turbo_batch_multiplier:.2f}x (+{adjustment_rate*100:.0f}%{vram_info})")
                    if self.turbo_epoch_multiplier != old_epoch:
                        changes.append(f"epochs {old_epoch:.2f}x → {self.turbo_epoch_multiplier:.2f}x")
                    print(f"   📈 GPU too low ({gpu_util:.1f}% < {self.gpu_target_util}%), increasing: {', '.join(changes)}")
        
        # Adjust based on VRAM usage
        if vram_used is not None:
            if vram_used > self.vram_limit_gb * 0.9:  # >90% of limit (7.2GB)
                # Reduce both multipliers to save memory
                self.turbo_batch_multiplier = max(2.0, self.turbo_batch_multiplier * (1 - 0.1))
                self.turbo_epoch_multiplier = max(1.5, self.turbo_epoch_multiplier * (1 - 0.1))
                print(warn(f"   [WARN] VRAM high ({vram_used:.2f}GB > {self.vram_limit_gb * 0.9:.1f}GB), reducing multipliers"))
            elif vram_used < self.vram_limit_gb * 0.5:  # <50% of limit (<4GB)
                # Can increase aggressively if GPU utilization allows
                if gpu_util is None or gpu_util < self.gpu_target_util:
                    old_batch = self.turbo_batch_multiplier
                    # If VRAM is very low (<20%), use even more aggressive adjustment
                    if vram_percent < 20:
                        vram_adjustment = 0.8  # 80% increase when VRAM is very low
                    elif vram_percent < 35:
                        vram_adjustment = adjustment_rate * 1.5  # 1.5x the normal rate
                    else:
                        vram_adjustment = adjustment_rate
                    
                    self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * (1 + vram_adjustment))
                    if self.turbo_batch_multiplier != old_batch:
                        print(f"   [MEM] VRAM low ({vram_used:.2f}GB, {vram_percent:.1f}%), increasing batch multiplier to {self.turbo_batch_multiplier:.2f}x (+{vram_adjustment*100:.0f}%)")
    
    def _archive_existing_models(self):
        """
        Archive existing model files to Archive folder before starting fresh training.
        This prevents accidentally reusing old (potentially bad) models.
        """
        import shutil
        from datetime import datetime
        
        # Find all model files (checkpoints, best_model, final_model)
        model_files = list(self.model_dir.glob("*.pt"))
        
        # Filter out files already in Archive
        model_files = [f for f in model_files if "Archive" not in str(f)]
        
        if not model_files:
            print("No existing models to archive (models directory is empty or already archived).")
            return
        
        # Create timestamped archive folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = self.archive_dir / f"archive_{timestamp}"
        archive_subdir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"ARCHIVING EXISTING MODELS")
        print(f"{'='*70}")
        print(f"Archive folder: {archive_subdir}")
        print(f"Found {len(model_files)} model file(s) to archive:")
        
        archived_count = 0
        for model_file in model_files:
            try:
                # Skip if already in Archive folder
                if "Archive" in str(model_file):
                    continue
                
                dest_path = archive_subdir / model_file.name
                shutil.move(str(model_file), str(dest_path))
                print(f"  [OK] Archived: {model_file.name}")
                archived_count += 1
            except Exception as e:
                print(warn(f"  [WARN] Failed to archive {model_file.name}: {e}"))
        
        print(f"\nArchived {archived_count} model file(s) to: {archive_subdir}")
        print(f"{'='*70}\n")
    
    def _archive_used_data_files(self):
        """
        Archive data files used during training to prevent reuse.
        Archives from both NT8 export folder and local data/raw folder.
        """
        import shutil
        from datetime import datetime
        
        if not hasattr(self, 'data_extractor') or not self.data_extractor:
            print("No data extractor found - skipping data file archiving.")
            return
        
        used_files = getattr(self.data_extractor, 'used_data_files', [])
        if not used_files:
            print("No data files tracked - skipping data file archiving.")
            return
        
        # Create timestamped archive folder (same timestamp as model archive if available)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archive NT8 export files
        nt8_archive_dir = None
        nt8_data_path = self.config.get("data", {}).get("nt8_data_path")
        if not nt8_data_path:
            # Try to get from settings
            try:
                settings_file = Path("settings.json")
                if settings_file.exists():
                    with open(settings_file, 'r') as f:
                        settings = json.load(f)
                        nt8_data_path = settings.get("nt8_data_path")
            except:
                pass
        
        # Also try common NT8 paths
        common_nt8_paths = [
            Path("C:/Users/schuo/Documents/NinjaTrader 8/export"),  # Current user path
            Path("C:/Users/sovan/Documents/NinjaTrader 8/export"),  # Previous user path
            Path.home() / "Documents" / "NinjaTrader 8" / "export",
            Path.home() / "Documents" / "NinjaTrader 8" / "Export",
        ]
        
        if nt8_data_path:
            common_nt8_paths.insert(0, Path(nt8_data_path))
        
        print(f"\n{'='*70}")
        print(f"ARCHIVING USED DATA FILES")
        print(f"{'='*70}")
        
        archived_count = 0
        
        # Deduplicate files by path (same file might be tracked multiple times)
        seen_files = set()
        unique_files = []
        for file_type, file_path in used_files:
            file_path_normalized = str(Path(file_path).resolve())
            if file_path_normalized not in seen_files:
                seen_files.add(file_path_normalized)
                unique_files.append((file_type, file_path))
        
        # Archive NT8 source files
        for file_type, file_path in unique_files:
            if file_type == "nt8_source":
                source_file = Path(file_path)
                if not source_file.exists():
                    continue
                
                # Find which NT8 path this file is in
                nt8_base_path = None
                source_file_resolved = source_file.resolve()
                source_file_str = str(source_file_resolved)
                for nt8_path in common_nt8_paths:
                    if not nt8_path.exists():
                        continue
                    try:
                        nt8_path_resolved = nt8_path.resolve()
                        nt8_path_str = str(nt8_path_resolved)
                        # Check if file is in this NT8 directory (using resolved paths)
                        if source_file_str.startswith(nt8_path_str) or str(source_file).startswith(str(nt8_path)):
                            nt8_base_path = nt8_path_resolved
                            break
                    except Exception:
                        # If resolve fails, try string comparison
                        if str(source_file).startswith(str(nt8_path)):
                            nt8_base_path = nt8_path
                            break
                
                if nt8_base_path:
                    # Create archive folder in NT8 export directory
                    nt8_archive_dir = nt8_base_path / "Archive" / f"archive_{timestamp}"
                    nt8_archive_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        dest_path = nt8_archive_dir / source_file.name
                        shutil.move(str(source_file), str(dest_path))
                        print(f"  [OK] Archived NT8 file: {source_file.name}")
                        archived_count += 1
                    except Exception as e:
                        print(warn(f"  [WARN] Failed to archive NT8 file {source_file.name}: {e}"))
        
        # Archive local data files
        local_archive_dir = Path("data/raw/Archive") / f"archive_{timestamp}"
        local_archive_dir.mkdir(parents=True, exist_ok=True)
        
        for file_type, file_path in unique_files:
            if file_type in ["local_file", "local_copy"]:
                local_file = Path(file_path)
                if not local_file.exists():
                    continue
                
                # Skip if already in Archive
                if "Archive" in str(local_file):
                    continue
                
                try:
                    dest_path = local_archive_dir / local_file.name
                    shutil.move(str(local_file), str(dest_path))
                    print(f"  [OK] Archived local file: {local_file.name}")
                    archived_count += 1
                except Exception as e:
                    print(warn(f"  [WARN] Failed to archive local file {local_file.name}: {e}"))
        
        if archived_count > 0:
            print(f"\nArchived {archived_count} data file(s)")
            if nt8_archive_dir:
                print(f"  NT8 Archive: {nt8_archive_dir}")
            print(f"  Local Archive: {local_archive_dir}")
        else:
            print("No data files were archived.")
        
        print(f"{'='*70}\n")
    
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
        self.data_extractor = extractor  # Store for archiving later
        instrument = self.config["environment"]["instrument"]
        timeframes = self.config["environment"]["timeframes"]
        trading_hours_cfg = self.config["environment"].get("trading_hours", {})
        trading_hours_manager = None
        if trading_hours_cfg.get("enabled"):
            trading_hours_manager = TradingHoursManager.from_dict(trading_hours_cfg)
            if trading_hours_manager.sessions:
                session_names = ", ".join(session.name for session in trading_hours_manager.sessions)
                print(f"  Trading hours enabled for sessions: {session_names}")
            else:
                print("  Trading hours enabled but no sessions defined (all times allowed).")
        
        # Try to load data
        try:
            self.multi_tf_data = extractor.load_multi_timeframe_data(
                instrument,
                timeframes,
                trading_hours=trading_hours_manager,
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
        print("Starting RL Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Timeframes: {self.config['environment']['timeframes']}")
        print(f"Instrument: {self.config['environment']['instrument']}")
        if hasattr(self, 'pretrained') and self.pretrained:
            print(success("  Pre-trained weights: ✅ Loaded (agent starts with learned patterns)"))
        else:
            print(info("  Pre-trained weights: ❌ None (starting from random initialization)"))
        print("="*60 + "\n")
        
        # Reset environment
        state, env_info = self.env.reset()  # Rename to avoid conflict with info() function
        
        episode_reward = 0
        episode_length = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Track initial performance for pre-training comparison
        initial_performance = {
            "first_episode_reward": None,
            "first_10_episodes_avg_reward": [],
            "first_10_episodes_avg_trades": []
        }
        best_mean_reward = float('-inf')
        
        # Progress bar
        pbar = tqdm(total=self.total_timesteps, desc="Training")
        
        # DEBUG: Log training loop start
        print(f"[DEBUG] Training loop starting: timestep={self.timestep}, total_timesteps={self.total_timesteps}, episode={self.episode}", flush=True)
        
        while self.timestep < self.total_timesteps:
            # Select action
            action, value, log_prob = self.agent.select_action(state)
            
            # Apply DecisionGate filtering if enabled (ensures consistency with live trading)
            if self.decision_gate:
                # Calculate RL confidence from action magnitude (proxy for confidence)
                rl_confidence = abs(float(action[0]))
                
                # Make decision through DecisionGate (RL-only mode during training)
                decision = self.decision_gate.make_decision(
                    rl_action=float(action[0]),
                    rl_confidence=rl_confidence,
                    reasoning_analysis=None,  # No reasoning during training
                    swarm_recommendation=None  # No swarm during training (unless enabled)
                )
                
                # Check if trade should execute based on DecisionGate filters
                if not self.decision_gate.should_execute(decision):
                    # Trade rejected by DecisionGate - use hold action (0.0)
                    action = np.array([0.0], dtype=np.float32)
                else:
                    # Trade approved - use DecisionGate's adjusted action (includes position sizing)
                    action = np.array([decision.action], dtype=np.float32)
            
            # Step environment
            try:
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
            except (IndexError, KeyError, Exception) as e:
                # CRITICAL FIX: Catch exceptions during step and terminate episode gracefully
                import traceback
                print(error(f"[ERROR] Exception in env.step() at episode {self.episode}, step {episode_length}: {e}"), flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                # Terminate episode on exception
                done = True
                terminated = True
                truncated = False
                reward = -1.0  # Negative reward for exception
                next_state = np.zeros(self.env.state_dim, dtype=np.float32)
                step_info = {"step": episode_length, "error": str(e)}
            
            # Update current episode trading metrics from step_info
            # ALWAYS update these values every step to ensure real-time updates
            if step_info:
                # Use episode_trades if available (episode-specific), otherwise fall back to trades (cumulative)
                self.current_episode_trades = step_info.get("episode_trades", step_info.get("trades", 0))
                self.current_episode_pnl = float(step_info.get("pnl", 0.0))
                self.current_episode_equity = float(step_info.get("equity", 0.0))
                self.current_episode_win_rate = float(step_info.get("win_rate", 0.0))
                self.current_episode_max_drawdown = float(step_info.get("max_drawdown", 0.0))
                
                # Track average win/loss and risk/reward ratio for profitability monitoring
                self.current_avg_win = float(step_info.get("avg_win", 0.0))
                self.current_avg_loss = float(step_info.get("avg_loss", 0.0))
                self.current_risk_reward_ratio = float(step_info.get("risk_reward_ratio", 0.0))
            else:
                # If step_info is missing, try to get values directly from environment
                # This ensures values are always updated even if step_info is incomplete
                if hasattr(self.env, 'episode_trades'):
                    self.current_episode_trades = self.env.episode_trades
                if hasattr(self.env, 'state') and self.env.state:
                    self.current_episode_pnl = float(self.env.state.total_pnl)
                    # Calculate equity from state
                    if hasattr(self.env, 'initial_capital'):
                        self.current_episode_equity = float(self.env.initial_capital + self.env.state.total_pnl)
                    # Calculate win rate from state
                    if self.env.state.trades_count > 0:
                        self.current_episode_win_rate = float(self.env.state.winning_trades / self.env.state.trades_count)
                if hasattr(self.env, 'max_drawdown'):
                    self.current_episode_max_drawdown = float(self.env.max_drawdown)
            
            # Real-time adaptive check for trading activity (lightweight, no full evaluation)
            if self.adaptive_trainer:
                adjustments = self.adaptive_trainer.check_trading_activity(
                    timestep=self.timestep,
                    episode=self.episode,
                    current_episode_trades=self.current_episode_trades,
                    current_episode_length=episode_length,
                    agent=self.agent
                )
                if adjustments:
                    # Adjustments already applied to agent, just log
                    print(f"[ADAPTIVE] Real-time adjustments applied at timestep {self.timestep:,}")
            
            # Store transition
            self.agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            self.current_episode_reward = episode_reward
            self.current_episode_length = episode_length
            
            # CRITICAL FIX: Increment timestep BEFORE checking done
            # This ensures timestep always increments even if episode ends immediately
            self.timestep += 1
            
            # DEBUG: Log timestep every 100 steps OR if episode ends immediately (to catch stuck timesteps)
            if self.timestep % 100 == 0 or (done and episode_length <= 1):
                print(f"[DEBUG] Timestep: {self.timestep:,}, Episode: {self.episode}, Episode Length: {episode_length}, Done: {done}", flush=True)
            
            pbar.update(1)
            
            # Update agent if buffer is full or episode ended
            if done or len(self.agent.states) >= self.config["model"]["n_steps"]:
                print(f"\n🔄 UPDATE TRIGGER CHECK: done={done}, buffer_size={len(self.agent.states)}, n_steps={self.config['model']['n_steps']}", flush=True)
                if len(self.agent.states) > 0:
                    print(f"✅ UPDATE WILL PROCEED: buffer has {len(self.agent.states)} samples", flush=True)
                    # Reload performance mode for dynamic adjustment
                    old_mode = self.performance_mode
                    self.performance_mode = self._load_performance_mode()
                    
                    # ALWAYS log current mode for verification (especially for Turbo)
                    if self.performance_mode == "turbo":
                        print(f"\n{'='*60}")
                        print(f"🔍 VERIFICATION: Performance mode = {self.performance_mode}")
                        print(f"🔍 Current Turbo multipliers: batch={self.turbo_batch_multiplier:.2f}x, epochs={self.turbo_epoch_multiplier:.2f}x")
                        print(f"{'='*60}")
                    
                    # Log if mode changed
                    if old_mode != self.performance_mode:
                        print(f"🔄 Training mode changed: {old_mode} → {self.performance_mode}")
                        print(f"   This will take effect in this update cycle")
                    
                    # Calculate dynamic batch size and epochs based on performance mode
                    base_batch_size = self.config["model"]["batch_size"]
                    base_n_epochs = self.config["model"].get("n_epochs", 10)
                    
                    # Get GPU stats before update (for Turbo mode adaptive adjustment)
                    gpu_stats_before = self._get_gpu_stats() if (self.device == "cuda" and self.performance_mode == "turbo") else None
                    
                    # VERIFICATION: Always log what mode we're in and what we're calculating
                    print(f"\n🔍 VERIFICATION LOG:")
                    print(f"   Mode detected: {self.performance_mode}")
                    print(f"   Base batch size: {base_batch_size}")
                    print(f"   Base epochs: {base_n_epochs}")
                    
                    if self.performance_mode == "turbo":
                        # Adaptive Turbo mode: Dynamically adjust based on GPU usage
                        # Target: 65% GPU utilization, <8GB VRAM
                        dynamic_batch_size = int(base_batch_size * self.turbo_batch_multiplier)
                        dynamic_n_epochs = int(base_n_epochs * self.turbo_epoch_multiplier)
                        
                        # Always log when turbo mode is active (for visibility)
                        print(f"\n🔥🔥🔥 TURBO MODE ACTIVE (ADAPTIVE) 🔥🔥🔥")
                        print(f"   Episode: {self.episode + 1}")
                        print(f"   Batch size: {dynamic_batch_size} ({self.turbo_batch_multiplier:.2f}x base: {base_batch_size})")
                        print(f"   Epochs: {dynamic_n_epochs} ({self.turbo_epoch_multiplier:.2f}x base: {base_n_epochs})")
                        print(f"   Target: {self.gpu_target_util}% GPU, <{self.vram_limit_gb}GB VRAM")
                        print(f"   🔍 Current multipliers: batch={self.turbo_batch_multiplier:.2f}x, epochs={self.turbo_epoch_multiplier:.2f}x")
                        if gpu_stats_before:
                            gpu_util = gpu_stats_before.get('utilization', 'N/A')
                            vram = gpu_stats_before.get('vram_used_gb', 'N/A')
                            print(f"   Before update: GPU {gpu_util}%, VRAM {vram:.2f}GB")
                        print(f"   💡 Adaptive adjustment based on GPU usage\n")
                    elif self.performance_mode == "performance":
                        # Use larger batch size and more epochs for faster training
                        dynamic_batch_size = base_batch_size * 2
                        dynamic_n_epochs = int(base_n_epochs * 1.5)
                    else:  # "quiet" mode - default
                        # Use configured batch size and epochs
                        dynamic_batch_size = base_batch_size
                        dynamic_n_epochs = base_n_epochs
                    
                    # Log update details before calling agent.update
                    print(f"\n{'='*60}")
                    print(f"📊 TRAINING UPDATE STARTING (Timestep: {self.timestep:,})")
                    print(f"{'='*60}")
                    print(f"   Mode: {self.performance_mode}")
                    if self.performance_mode == "turbo":
                        print(f"   🔥 Turbo multipliers: batch={self.turbo_batch_multiplier:.2f}x, epochs={self.turbo_epoch_multiplier:.2f}x")
                    print(f"   Batch size: {dynamic_batch_size} (base: {base_batch_size}, multiplier: {dynamic_batch_size/base_batch_size:.2f}x)")
                    print(f"   Epochs: {dynamic_n_epochs} (base: {base_n_epochs}, multiplier: {dynamic_n_epochs/base_n_epochs:.2f}x)")
                    print(f"   Buffer size: {len(self.agent.states)} samples")
                    print(f"   ⏱️  Update may take 1-5 minutes in Turbo mode - timestep will remain at {self.timestep:,} until update completes")
                    print(f"{'='*60}\n")
                    
                    # Update agent (GPU-intensive operation)
                    import threading
                    import time
                    
                    # Initialize VRAM tracking variables
                    vram_before = 0.0
                    vram_cached_before = 0.0
                    gpu_start = time.time()
                    buffer_size = len(self.agent.states)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()  # Sync before timing
                        batches_per_epoch = (buffer_size + dynamic_batch_size - 1) // dynamic_batch_size
                        total_batches = dynamic_n_epochs * batches_per_epoch
                        
                        # Get VRAM before update
                        vram_before = torch.cuda.memory_allocated(0) / 1e9
                        vram_cached_before = torch.cuda.memory_reserved(0) / 1e9
                        
                        print(f"🚀 Starting GPU update: {total_batches} total batches ({batches_per_epoch} batches × {dynamic_n_epochs} epochs)")
                        print(f"   Buffer size: {buffer_size}, Batch size: {dynamic_batch_size}, Epochs: {dynamic_n_epochs}")
                        print(f"   VRAM BEFORE update: {vram_before:.3f}GB allocated, {vram_cached_before:.3f}GB reserved")
                        print(f"   ⚡⚡⚡ GPU SHOULD SPIKE NOW - Check nvidia-smi or ./check_gpu.sh ⚡⚡⚡")
                    
                    # For Turbo mode: Sample GPU during update to capture peak utilization
                    peak_gpu_util = None
                    peak_gpu_stats = None
                    if self.device == "cuda" and self.performance_mode == "turbo":
                        sampling_active = threading.Event()
                        sampling_active.set()
                        
                        def sample_gpu_during_update():
                            """Background thread to sample GPU during update"""
                            nonlocal peak_gpu_util, peak_gpu_stats
                            max_util = 0
                            best_stats = None
                            while sampling_active.is_set():
                                stats = self._get_gpu_stats()
                                if stats and stats.get('utilization') is not None:
                                    util = stats['utilization']
                                    if util > max_util:
                                        max_util = util
                                        best_stats = stats.copy()  # Make a copy to preserve peak stats
                                time.sleep(0.3)  # Sample every 300ms
                            peak_gpu_util = max_util if max_util > 0 else None
                            peak_gpu_stats = best_stats if best_stats else None
                            # Ensure peak_gpu_stats has the peak utilization value
                            if peak_gpu_stats and peak_gpu_util is not None:
                                peak_gpu_stats['utilization'] = peak_gpu_util
                        
                        # Start GPU sampling thread
                        sampling_thread = threading.Thread(target=sample_gpu_during_update, daemon=True)
                        sampling_thread.start()
                    
                    metrics = self.agent.update(
                        n_epochs=dynamic_n_epochs,
                        batch_size=dynamic_batch_size,
                        scaler=self.scaler,
                        autocast=self.autocast
                    )
                    
                    # Stop GPU sampling and get peak
                    if self.device == "cuda" and self.performance_mode == "turbo":
                        sampling_active.clear()
                        sampling_thread.join(timeout=1.0)  # Wait for thread to finish
                    
                    # Log GPU usage after update and adjust Turbo mode if needed
                    if self.device == "cuda":
                        torch.cuda.synchronize()  # Sync after GPU work
                        gpu_elapsed = time.time() - gpu_start
                        gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9
                        gpu_mem_cached = torch.cuda.memory_reserved(0) / 1e9
                        
                        # Get GPU stats after update (for adaptive Turbo mode)
                        # Use peak stats if available (sampled during update), otherwise get current stats
                        if self.performance_mode == "turbo":
                            # Prefer peak stats captured during update
                            gpu_stats_after = peak_gpu_stats if peak_gpu_stats else self._get_gpu_stats()
                            if peak_gpu_util is not None:
                                print(f"   📊 Peak GPU during update: {peak_gpu_util:.1f}% (sampled during update)")
                        else:
                            gpu_stats_after = None
                        
                        # Always log for Turbo mode, or every 10 episodes otherwise
                        if self.performance_mode == "turbo" or self.episode % 10 == 0:
                            if self.device == "cuda" and vram_before > 0:
                                vram_change = gpu_mem_used - vram_before
                                vram_change_pct = (vram_change / vram_before * 100)
                                print(f"\n✅ GPU Update Complete: {gpu_elapsed:.2f}s")
                                print(f"   VRAM: {vram_before:.3f}GB → {gpu_mem_used:.3f}GB ({vram_change:+.3f}GB, {vram_change_pct:+.1f}%)")
                                print(f"   VRAM Reserved: {vram_cached_before:.3f}GB → {gpu_mem_cached:.3f}GB")
                                if abs(vram_change) < 0.01:
                                    print(f"   ⚠️  WARNING: VRAM barely changed ({vram_change:+.3f}GB) - batch size might be too small or model is too efficient")
                            else:
                                print(f"\n✅ GPU Update Complete: {gpu_elapsed:.2f}s")
                                print(f"   Memory: {gpu_mem_used:.2f}GB used / {gpu_mem_cached:.2f}GB cached")
                            if gpu_elapsed < 1.0:
                                print(f"   ⚠️  WARNING: Update completed in {gpu_elapsed:.2f}s - this is very fast, might indicate small batch size or model")
                            
                            # Show adaptive Turbo stats
                            if self.performance_mode == "turbo":
                                if gpu_stats_after:
                                    gpu_util = peak_gpu_util if peak_gpu_util is not None else gpu_stats_after.get('utilization')
                                    vram_used = gpu_stats_after.get('vram_used_gb')
                                    
                                    if gpu_util is not None:
                                        stat_source = "peak during update" if peak_gpu_util else "after update"
                                        vram_percent = (vram_used / self.vram_limit_gb * 100) if self.vram_limit_gb > 0 else 0
                                        print(f"   📊 GPU {stat_source}: {gpu_util:.1f}% (target: {self.gpu_target_util}%), VRAM {vram_used:.2f}GB ({vram_percent:.1f}% of limit)")
                                        
                                        # If VRAM is very low, be extra aggressive regardless of GPU
                                        old_batch_mult = self.turbo_batch_multiplier
                                        
                                        # Check most aggressive first (VRAM < 15%)
                                        if vram_percent < 15 and gpu_util < self.gpu_target_util:
                                            print(f"   🚀🚀 VRAM is EXTREMELY low ({vram_percent:.1f}%), forcing aggressive increase!")
                                            old_epoch_mult = self.turbo_epoch_multiplier
                                            # Triple the batch multiplier if VRAM is extremely low
                                            self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 3.0)
                                            # Also increase epochs aggressively (more work per update = more GPU time)
                                            self.turbo_epoch_multiplier = min(15.0, self.turbo_epoch_multiplier * 2.5)
                                            if self.turbo_batch_multiplier != old_batch_mult:
                                                print(f"   📈 TRIPLING batch multiplier: {old_batch_mult:.2f}x → {self.turbo_batch_multiplier:.2f}x")
                                            else:
                                                print(f"   ⚠️  Batch multiplier already at max ({self.max_batch_multiplier}x)")
                                            if self.turbo_epoch_multiplier != old_epoch_mult:
                                                print(f"   📈 DOUBLING epoch multiplier: {old_epoch_mult:.2f}x → {self.turbo_epoch_multiplier:.2f}x")
                                        elif vram_percent < 20 and gpu_util < self.gpu_target_util:
                                            print(f"   🚀 VRAM is very low ({vram_percent:.1f}%), making aggressive increase!")
                                            old_epoch_mult = self.turbo_epoch_multiplier
                                            # Double the batch multiplier if VRAM is this low
                                            self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 2.0)
                                            # Also increase epochs (more work per update)
                                            self.turbo_epoch_multiplier = min(12.0, self.turbo_epoch_multiplier * 2.0)
                                            if self.turbo_batch_multiplier != old_batch_mult:
                                                print(f"   📈 DOUBLING batch multiplier: {old_batch_mult:.2f}x → {self.turbo_batch_multiplier:.2f}x")
                                            else:
                                                print(f"   ⚠️  Batch multiplier already at max ({self.max_batch_multiplier}x)")
                                            if self.turbo_epoch_multiplier != old_epoch_mult:
                                                print(f"   📈 Increasing epoch multiplier: {old_epoch_mult:.2f}x → {self.turbo_epoch_multiplier:.2f}x (+50%)")
                                        else:
                                            # Normal adjustment based on GPU
                                            self._adjust_turbo_multipliers(gpu_stats_before, gpu_stats_after)
                                        
                                        # Log if multiplier changed
                                        if self.turbo_batch_multiplier != old_batch_mult:
                                            print(f"   🔄 Next update will use batch multiplier: {self.turbo_batch_multiplier:.2f}x (was {old_batch_mult:.2f}x)")
                                        elif vram_percent < 20:
                                            print(f"   ⚠️  Multiplier unchanged but VRAM is low - check if max limit reached")
                                        
                                        if gpu_util <= self.gpu_target_util + 5 and vram_used < self.vram_limit_gb:
                                            print(f"   ✅ GPU usage within target range!")
                                    else:
                                        print(f"   📊 After update: VRAM {vram_used:.2f}GB (limit: {self.vram_limit_gb}GB)")
                                        vram_percent_fallback = (vram_used / self.vram_limit_gb * 100) if self.vram_limit_gb > 0 else 0
                                        old_batch = self.turbo_batch_multiplier
                                        
                                        if vram_percent_fallback < 15:  # VRAM < 15% (1.2GB) - extremely low
                                            print(f"   🚀🚀 Can't read GPU, but VRAM is EXTREMELY low ({vram_percent_fallback:.1f}%)!")
                                            print(f"   🚀 TRIPLING batch multiplier anyway (VRAM headroom available)")
                                            self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 3.0)
                                            if self.turbo_batch_multiplier != old_batch:
                                                print(f"   📈 Increasing batch multiplier: {old_batch:.2f}x → {self.turbo_batch_multiplier:.2f}x (TRIPLED)")
                                        elif vram_percent_fallback < 20:  # VRAM < 20% (1.6GB)
                                            print(f"   ⚠️  Can't read GPU, but VRAM is very low ({vram_percent_fallback:.1f}%)")
                                            print(f"   🚀 DOUBLING batch multiplier anyway (VRAM headroom available)")
                                            self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 2.0)
                                            if self.turbo_batch_multiplier != old_batch:
                                                print(f"   📈 Increasing batch multiplier: {old_batch:.2f}x → {self.turbo_batch_multiplier:.2f}x (DOUBLED)")
                                        elif vram_used < self.vram_limit_gb * 0.3:  # VRAM < 30% (2.4GB)
                                            print(f"   ⚠️  Can't read GPU, but VRAM is low ({vram_used:.2f}GB)")
                                            print(f"   🚀 Making aggressive increase anyway (VRAM headroom available)")
                                            self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 1.5)
                                            if self.turbo_batch_multiplier != old_batch:
                                                print(f"   📈 Increasing batch multiplier: {old_batch:.2f}x → {self.turbo_batch_multiplier:.2f}x (+50%)")
                                        elif vram_used < self.vram_limit_gb:
                                            print(f"   ✅ VRAM within limit")
                                        else:
                                            self._adjust_turbo_multipliers(gpu_stats_before, gpu_stats_after)
                                else:
                                    # No GPU stats available - if VRAM is low, increase aggressively
                                    vram_used = gpu_mem_used
                                    if vram_used < self.vram_limit_gb * 0.3:  # < 2.4GB
                                        print(f"   ⚠️  GPU stats unavailable, but VRAM is very low ({vram_used:.2f}GB)")
                                        print(f"   🚀 Making aggressive increase (VRAM headroom available)")
                                        old_batch = self.turbo_batch_multiplier
                                        self.turbo_batch_multiplier = min(self.max_batch_multiplier, self.turbo_batch_multiplier * 1.5)
                                        if self.turbo_batch_multiplier != old_batch:
                                            print(f"   📈 Increasing batch multiplier: {old_batch:.2f}x → {self.turbo_batch_multiplier:.2f}x (+50%)")
                    
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
                        
                        # Log pre-training status (if applicable)
                        if hasattr(self, 'pretrained') and self.pretrained and self.timestep < 1000:
                            # Track early performance to compare with/without pre-training
                            self.writer.add_scalar("pretraining/used", 1.0, self.timestep)
            
            # Handle episode end
            if done:
                # IMPORTANT: Use current_episode_trades which is updated during the loop
                # This is more reliable than step_info which might not have the latest value
                episode_trades = self.current_episode_trades
                
                # Fallback to step_info or environment if current_episode_trades is 0 (shouldn't happen)
                if episode_trades == 0:
                    episode_trades_from_env = getattr(self.env, 'episode_trades', 0)
                    episode_trades = step_info.get("episode_trades", episode_trades_from_env) if step_info else episode_trades_from_env
                    if episode_trades == 0:
                        # Final fallback to cumulative trades (shouldn't happen, but just in case)
                        episode_trades = step_info.get("trades", 0) if step_info else 0
                
                episode_pnl = float(step_info.get("pnl", 0.0)) if step_info else 0.0
                episode_equity = float(step_info.get("equity", 0.0)) if step_info else 0.0
                episode_win_rate = float(step_info.get("win_rate", 0.0)) if step_info else 0.0
                episode_max_drawdown = float(step_info.get("max_drawdown", 0.0)) if step_info else 0.0
                
                self.episode += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Track initial performance for pre-training comparison
                if self.episode == 1:
                    initial_performance["first_episode_reward"] = episode_reward
                if self.episode <= 10:
                    initial_performance["first_10_episodes_avg_reward"].append(episode_reward)
                    initial_performance["first_10_episodes_avg_trades"].append(episode_trades)
                
                # Log initial performance metrics (first 10 episodes)
                if self.episode == 10 and hasattr(self, 'pretrained') and self.pretrained and self.writer:
                    avg_reward = np.mean(initial_performance["first_10_episodes_avg_reward"])
                    avg_trades = np.mean(initial_performance["first_10_episodes_avg_trades"])
                    self.writer.add_scalar("pretraining/initial_avg_reward", avg_reward, self.timestep)
                    self.writer.add_scalar("pretraining/initial_avg_trades", avg_trades, self.timestep)
                    from src.utils.colors import info as info_color
                    print(info_color(f"[PRETRAIN] Initial performance (first 10 episodes): Avg reward={avg_reward:.2f}, Avg trades={avg_trades:.1f}"))
                
                # Store trading metrics for this episode
                self.episode_trades.append(episode_trades)
                self.episode_pnls.append(episode_pnl)
                self.episode_equities.append(episode_equity)
                self.episode_win_rates.append(episode_win_rate)
                self.episode_max_drawdowns.append(episode_max_drawdown)
                
                # CRITICAL FIX: Update adaptive trainer's consecutive_no_trade_episodes counter
                if self.adaptive_trainer:
                    if episode_trades == 0:
                        self.adaptive_trainer.consecutive_no_trade_episodes += 1
                        print(f"[ADAPTIVE] Episode {self.episode} completed with 0 trades. Consecutive no-trade episodes: {self.adaptive_trainer.consecutive_no_trade_episodes}")
                        
                        # CRITICAL: Trigger immediate adaptive check when episode completes with no trades
                        # This bypasses timestep checks for persistent no-trade conditions
                        adjustments = self.adaptive_trainer.check_trading_activity(
                            timestep=self.timestep,
                            episode=self.episode,
                            current_episode_trades=0,
                            current_episode_length=episode_length,
                            agent=self.agent
                        )
                        if adjustments:
                            print(f"[ADAPTIVE] ⚡ IMMEDIATE adjustment triggered after episode {self.episode} with no trades!")
                            for key, value in adjustments.items():
                                if isinstance(value, dict):
                                    for sub_key, sub_value in value.items():
                                        if isinstance(sub_value, dict):
                                            print(f"   {key}.{sub_key}: {sub_value.get('old', 'N/A')} -> {sub_value.get('new', 'N/A')}")
                    else:
                        if self.adaptive_trainer.consecutive_no_trade_episodes > 0:
                            print(f"[ADAPTIVE] Episode {self.episode} had {episode_trades} trades. Resetting consecutive no-trade counter (was {self.adaptive_trainer.consecutive_no_trade_episodes})")
                        self.adaptive_trainer.consecutive_no_trade_episodes = 0
                
                # NEW: Quick episode-level check for recent negative trend (every episode)
                if self.adaptive_trainer and len(self.episode_pnls) >= 10:
                    recent_pnls = self.episode_pnls[-10:]
                    recent_mean_pnl = sum(recent_pnls) / len(recent_pnls)
                    
                    # If negative trend for 10+ episodes, trigger quick adjustment
                    if recent_mean_pnl < 0:
                        # Get recent trades from journal if available (for better analysis)
                        recent_trades_data = None
                        if self.journal_integration and hasattr(self.journal_integration, 'journal'):
                            try:
                                # Get last 20 trades from journal
                                recent_trades = self.journal_integration.journal.get_recent_trades(limit=20)
                                if recent_trades:
                                    recent_trades_data = [
                                        {
                                            "pnl": float(row.get("pnl", 0)),
                                            "net_pnl": float(row.get("net_pnl", 0)),
                                            "strategy": row.get("strategy", "RL"),
                                            "entry_price": float(row.get("entry_price", 0)),
                                            "exit_price": float(row.get("exit_price", 0)),
                                            "is_win": bool(row.get("is_win", False))
                                        }
                                        for row in recent_trades
                                    ]
                            except Exception as e:
                                # If journal access fails, continue without it
                                pass
                        
                        # Quick adjustment without full evaluation (ENHANCED: now uses journal data)
                        # Calculate total trades in recent episodes
                        recent_total_trades = sum(self.episode_trades[-10:]) if len(self.episode_trades) >= 10 else sum(self.episode_trades)
                        
                        quick_adjustments = self.adaptive_trainer.quick_adjust_for_negative_trend(
                            recent_mean_pnl=recent_mean_pnl,
                            recent_win_rate=sum(self.episode_win_rates[-10:]) / 10 if len(self.episode_win_rates) >= 10 else 0.0,
                            agent=self.agent,
                            recent_trades_data=recent_trades_data,  # NEW: Pass journal data
                            recent_total_trades=recent_total_trades  # NEW: Pass total trades count
                        )
                        if quick_adjustments:
                            print(f"[ADAPT] Quick adjustment triggered: {len(quick_adjustments)} adjustments")
                            for key, value in quick_adjustments.items():
                                if isinstance(value, dict):
                                    for sub_key, sub_value in value.items():
                                        if isinstance(sub_value, dict):
                                            print(f"   {key}.{sub_key}: {sub_value.get('old', 'N/A')} -> {sub_value.get('new', 'N/A')} ({sub_value.get('reason', '')})")
                                else:
                                    print(f"   {key}: {value}")
                
                # Update cumulative totals (estimate wins/losses from win rate)
                old_total_trades = self.total_trades
                self.total_trades += episode_trades
                if episode_trades > 0:
                    estimated_wins = int(episode_trades * episode_win_rate)
                    estimated_losses = episode_trades - estimated_wins
                    self.total_winning_trades += estimated_wins
                    self.total_losing_trades += estimated_losses
                else:
                    print(warn(f"[WARN] Episode {self.episode} had 0 trades! current_episode_trades={self.current_episode_trades}"), flush=True)
                sys.stdout.flush()
                
                # Log episode metrics
                if self.writer:
                    self.writer.add_scalar("episode/reward", episode_reward, self.episode)
                    self.writer.add_scalar("episode/length", episode_length, self.episode)
                    self.writer.add_scalar("episode/trades", episode_trades, self.episode)
                    self.writer.add_scalar("episode/pnl", episode_pnl, self.episode)
                    self.writer.add_scalar("episode/equity", episode_equity, self.episode)
                    self.writer.add_scalar("episode/win_rate", episode_win_rate, self.episode)
                    self.writer.add_scalar("episode/max_drawdown", episode_max_drawdown, self.episode)
                
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
                state, env_info = self.env.reset(options={"episode": self.episode})  # Rename to avoid conflict with info() function
                # Update episode number in environment for journaling
                if hasattr(self.env, '_current_episode'):
                    self.env._current_episode = self.episode
                episode_reward = 0
                episode_length = 0
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
                self.current_episode_trades = 0
                self.current_episode_pnl = 0.0
                self.current_episode_equity = 0.0
                self.current_episode_win_rate = 0.0
                self.current_episode_max_drawdown = 0.0
                self.current_avg_win = 0.0
                self.current_avg_loss = 0.0
                self.current_risk_reward_ratio = 0.0
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
                    self.episode_lengths,
                    episode_pnls=self.episode_pnls,
                    episode_equities=self.episode_equities,
                    episode_win_rates=self.episode_win_rates
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
            
            # Evaluation and adaptive adjustment
            # CRITICAL FIX: Also check for evaluation if timestep is stuck at 0
            # Use episode-based evaluation when timestep is 0
            # FIX: Check both eval_freq (config) and adaptive_trainer.should_evaluate() to ensure evaluations happen
            should_eval = False
            if self.timestep > 0:
                # Check if timestep matches eval_freq OR if adaptive trainer says we should evaluate
                should_eval = (self.timestep % self.eval_freq == 0) or \
                             (self.adaptive_trainer and self.adaptive_trainer.should_evaluate(self.timestep))
            elif self.timestep == 0 and self.adaptive_trainer:
                # If timestep is stuck at 0, use episode-based evaluation (every 10 episodes)
                should_eval = (self.episode > 0 and self.episode % 10 == 0)
            
            if should_eval:
                if self.adaptive_trainer and (self.timestep > 0 and self.adaptive_trainer.should_evaluate(self.timestep)) or \
                   (self.timestep == 0 and self.adaptive_trainer):
                    # Use adaptive trainer for intelligent evaluation and adjustment
                    checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
                    if not checkpoint_path.exists():
                        # Create temporary checkpoint for evaluation
                        self.agent.save_with_training_state(
                            str(checkpoint_path),
                            self.timestep,
                            self.episode,
                            self.episode_rewards,
                            self.episode_lengths
                        )
                    
                    mean_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else (np.mean(self.episode_rewards) if self.episode_rewards else 0)
                    # Get policy loss from agent if available
                    policy_loss = getattr(self.agent, 'last_policy_loss', None)
                    
                    result = self.adaptive_trainer.evaluate_and_adapt(
                        model_path=str(checkpoint_path),
                        timestep=self.timestep,
                        episode=self.episode,
                        mean_reward=mean_reward,
                        agent=self.agent,
                        policy_loss=policy_loss  # Pass policy loss for convergence detection
                    )
                    
                    # Apply adjustments
                    if result.get("adjustments"):
                        adjustments = result["adjustments"]
                        if "entropy_coef" in adjustments:
                            print(f"✅ Applied entropy_coef adjustment: {adjustments['entropy_coef']['new']:.4f}")
                        if "inaction_penalty" in adjustments:
                            print(f"✅ Applied inaction_penalty adjustment: {adjustments['inaction_penalty']['new']:.6f}")
                        if "learning_rate" in adjustments:
                            print(f"✅ Applied learning_rate adjustment: {adjustments['learning_rate']['new']:.6f}")
                    
                    # Check if training should be paused
                    if self.adaptive_trainer.is_training_paused():
                        print("\n" + "="*70)
                        print("[PAUSED] Training paused by adaptive learning system")
                        print("="*70)
                        print(f"Reason: {self.adaptive_trainer.get_pause_reason()}")
                        print("\nSaving final checkpoint...")
                        # Save checkpoint
                        checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
                        self.agent.save_with_training_state(
                            str(checkpoint_path),
                            self.timestep,
                            self.episode,
                            self.episode_rewards,
                            self.episode_lengths
                        )
                        print(f"✅ Checkpoint saved: {checkpoint_path}")
                        print("\n[ACTION REQUIRED] Review performance and adjust parameters before resuming.")
                        print("Training stopped.")
                        print("="*70 + "\n")
                        break  # Exit training loop
                    
                    # Auto-save on improvement
                    if result.get("should_save") and result.get("improvement"):
                        best_path = self.model_dir / "best_model.pt"
                        self.agent.save_with_training_state(
                            str(best_path),
                            self.timestep,
                            self.episode,
                            self.episode_rewards,
                            self.episode_lengths
                        )
                        print(f"💾 Auto-saved best model (improvement: {result['improvement']*100:.1f}%)")
                    
                    # Early stopping check (after adaptive evaluation)
                    if self.early_stopping_enabled:
                        # Get current metric from evaluation result or episode metrics
                        current_metric = None
                        if self.early_stopping_metric_name == "mean_reward":
                            # Use mean reward from recent episodes
                            current_metric = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else (np.mean(self.episode_rewards) if self.episode_rewards else float('-inf'))
                        elif self.early_stopping_metric_name == "win_rate":
                            # Use win rate from recent episodes
                            if len(self.episode_win_rates) >= 10:
                                current_metric = np.mean(self.episode_win_rates[-10:])
                            elif self.episode_win_rates:
                                current_metric = np.mean(self.episode_win_rates)
                            else:
                                current_metric = 0.0
                        
                        if current_metric is not None:
                            # Check if current metric is better than best (with min_delta threshold)
                            improvement = current_metric - self.early_stopping_best_metric
                            if improvement >= self.early_stopping_min_delta:
                                # New best metric found
                                self.early_stopping_best_metric = current_metric
                                self.early_stopping_last_improvement = self.timestep
                                print(f"📈 Early stopping: New best {self.early_stopping_metric_name} = {current_metric:.4f} (improvement: {improvement:.4f})")
                            else:
                                # No improvement
                                steps_since_improvement = self.timestep - self.early_stopping_last_improvement
                                if steps_since_improvement >= self.early_stopping_patience:
                                    print("\n" + "="*70)
                                    print("[EARLY STOPPING] Training stopped - no improvement detected")
                                    print("="*70)
                                    print(f"Best {self.early_stopping_metric_name}: {self.early_stopping_best_metric:.4f}")
                                    print(f"Current {self.early_stopping_metric_name}: {current_metric:.4f}")
                                    print(f"Steps since last improvement: {steps_since_improvement:,} / {self.early_stopping_patience:,}")
                                    print(f"Min delta required: {self.early_stopping_min_delta:.4f}")
                                    print("\nSaving final checkpoint...")
                                    # Save checkpoint
                                    checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
                                    self.agent.save_with_training_state(
                                        str(checkpoint_path),
                                        self.timestep,
                                        self.episode,
                                        self.episode_rewards,
                                        self.episode_lengths
                                    )
                                    print(f"✅ Checkpoint saved: {checkpoint_path}")
                                    print("\nTraining stopped to prevent overfitting.")
                                    print("="*70 + "\n")
                                    break  # Exit training loop
                                else:
                                    remaining = self.early_stopping_patience - steps_since_improvement
                                    if steps_since_improvement % 10000 == 0:  # Print every 10k steps
                                        print(f"⏳ Early stopping: No improvement for {steps_since_improvement:,} steps ({remaining:,} remaining)")
                else:
                    # Standard evaluation
                    eval_metrics = self._evaluate()
                    
                    # Early stopping check (after standard evaluation)
                    if self.early_stopping_enabled and eval_metrics:
                        # Get current metric from evaluation
                        current_metric = None
                        if self.early_stopping_metric_name == "mean_reward":
                            current_metric = eval_metrics.get("mean_reward", np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else (np.mean(self.episode_rewards) if self.episode_rewards else float('-inf')))
                        elif self.early_stopping_metric_name == "win_rate":
                            current_metric = eval_metrics.get("win_rate", np.mean(self.episode_win_rates[-10:]) if len(self.episode_win_rates) >= 10 else (np.mean(self.episode_win_rates) if self.episode_win_rates else 0.0))
                        
                        if current_metric is not None:
                            # Check if current metric is better than best (with min_delta threshold)
                            improvement = current_metric - self.early_stopping_best_metric
                            if improvement >= self.early_stopping_min_delta:
                                # New best metric found
                                self.early_stopping_best_metric = current_metric
                                self.early_stopping_last_improvement = self.timestep
                                print(f"📈 Early stopping: New best {self.early_stopping_metric_name} = {current_metric:.4f} (improvement: {improvement:.4f})")
                            else:
                                # No improvement
                                steps_since_improvement = self.timestep - self.early_stopping_last_improvement
                                if steps_since_improvement >= self.early_stopping_patience:
                                    print("\n" + "="*70)
                                    print("[EARLY STOPPING] Training stopped - no improvement detected")
                                    print("="*70)
                                    print(f"Best {self.early_stopping_metric_name}: {self.early_stopping_best_metric:.4f}")
                                    print(f"Current {self.early_stopping_metric_name}: {current_metric:.4f}")
                                    print(f"Steps since last improvement: {steps_since_improvement:,} / {self.early_stopping_patience:,}")
                                    print(f"Min delta required: {self.early_stopping_min_delta:.4f}")
                                    print("\nSaving final checkpoint...")
                                    # Save checkpoint
                                    checkpoint_path = self.model_dir / f"checkpoint_{self.timestep}.pt"
                                    self.agent.save_with_training_state(
                                        str(checkpoint_path),
                                        self.timestep,
                                        self.episode,
                                        self.episode_rewards,
                                        self.episode_lengths
                                    )
                                    print(f"✅ Checkpoint saved: {checkpoint_path}")
                                    print("\nTraining stopped to prevent overfitting.")
                                    print("="*70 + "\n")
                                    break  # Exit training loop
        
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
        
        # Archive data files used during training
        self._archive_used_data_files()
        
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
            action_threshold=self.config["environment"].get("action_threshold", 0.05),  # Configurable action threshold
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
                state, reward, terminated, truncated, eval_info = eval_env.step(action)  # Rename to avoid conflict with info() function
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_pnls.append(eval_info.get("pnl", 0))
        
        mean_reward = np.mean(eval_rewards)
        mean_pnl = np.mean(eval_pnls)
        
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", mean_reward, self.timestep)
            self.writer.add_scalar("eval/mean_pnl", mean_pnl, self.timestep)
        
        print(f"\n📊 Evaluation @ step {self.timestep}: "
              f"Mean reward: {mean_reward:.2f}, Mean PnL: ${mean_pnl:.2f}")
        
        # Return metrics for early stopping
        return {
            "mean_reward": mean_reward,
            "mean_pnl": mean_pnl
        }
    
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
        default="configs/train_config_full.yaml",
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
    trainer = Trainer(config, checkpoint_path=args.checkpoint, config_path=args.config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()

