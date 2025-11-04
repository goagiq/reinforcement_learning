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
import subprocess
import re

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
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✅ Using GPU: {gpu_name} (CUDA {cuda_version})")
                    print(f"   GPU Memory: {gpu_memory:.1f} GB total")
                    print(f"   Device will be used for: Actor/Critic networks, tensor operations")
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
        import time
        data_load_start = time.time()
        print("Loading data...")
        print(f"  Instrument: {config['environment']['instrument']}")
        print(f"  Timeframes: {config['environment']['timeframes']}")
        try:
            self._load_data()
            data_load_elapsed = time.time() - data_load_start
            print(f"✅ Data loaded successfully (took {data_load_elapsed:.1f}s)")
        except Exception as e:
            data_load_elapsed = time.time() - data_load_start
            print(f"❌ Error loading data after {data_load_elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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
        
        # Check if we're resuming from checkpoint - if so, load architecture from checkpoint
        checkpoint_hidden_dims = None
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
                    
                    if checkpoint_hidden_dims:
                        print(f"📐 Found architecture in checkpoint: hidden_dims={checkpoint_hidden_dims}")
                        print(f"   Will use this architecture instead of config to match checkpoint")
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
                                else:
                                    print(f"⚠️  Could not infer architecture, will use config/default")
                            else:
                                print(f"⚠️  Could not find feature layers in checkpoint")
                except Exception as e:
                    print(f"⚠️  Could not read checkpoint architecture: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"   Will use config/default architecture")
        
        # Get network architecture - use checkpoint if available, else config, else default
        if checkpoint_hidden_dims:
            hidden_dims = checkpoint_hidden_dims
            print(f"   Using architecture from checkpoint: {hidden_dims}")
        else:
            hidden_dims = model_config.get("hidden_dims", [256, 256, 128])
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
        
        # Adaptive Turbo mode: Track multipliers for dynamic adjustment
        # Target: 65% GPU utilization, <8GB VRAM
        # Start at MAXIMUM since VRAM is only 11-14% - model is severely underutilized
        self.turbo_batch_multiplier = 50.0  # Start at MAXIMUM (50x = 6400 batch size for base 128)
        self.turbo_epoch_multiplier = 8.0   # Start VERY aggressive (8x = 240 epochs with base 30)
        self.gpu_target_util = 65.0  # Target GPU utilization %
        self.vram_limit_gb = 8.0    # VRAM limit in GB
        self.max_batch_multiplier = 100.0  # Increased max multiplier (for very small models with lots of VRAM)
        
        if self.performance_mode == "turbo":
            print(f"⚙️  Performance mode: {self.performance_mode} (ADAPTIVE GPU UTILIZATION)")
            print(f"   Adaptive Turbo: Targeting {self.gpu_target_util}% GPU, <{self.vram_limit_gb}GB VRAM")
            print(f"   Starting: {self.turbo_batch_multiplier}x batch, {self.turbo_epoch_multiplier}x epochs")
        else:
            print(f"⚙️  Performance mode: {self.performance_mode}")
        
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
                timestep, episode, rewards, lengths = self.agent.load_with_training_state(str(checkpoint_path))
                checkpoint_timestep = timestep
                self.timestep = timestep
                self.episode = episode
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                print(f"✅ Resume: timestep={timestep}, episode={episode}, rewards={len(rewards)}")
                
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
                        print(f"✅ TURBO MODE DETECTED in settings.json (turbo_training_mode: {turbo_enabled})")
                        return "turbo"
                    else:
                        print(f"📊 Performance mode: {perf_mode} (turbo_training_mode: {turbo_enabled})")
                        return perf_mode
            except Exception as e:
                print(f"⚠️  Error loading performance mode: {e}")
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
                print(f"   📉 GPU too high ({gpu_util:.1f}% > {self.gpu_target_util}%), reducing batch multiplier to {self.turbo_batch_multiplier:.2f}x")
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
                print(f"   ⚠️  VRAM high ({vram_used:.2f}GB > {self.vram_limit_gb * 0.9:.1f}GB), reducing multipliers")
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
                        print(f"   💾 VRAM low ({vram_used:.2f}GB, {vram_percent:.1f}%), increasing batch multiplier to {self.turbo_batch_multiplier:.2f}x (+{vram_adjustment*100:.0f}%)")
    
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
                    print(f"📊 TRAINING UPDATE STARTING")
                    print(f"{'='*60}")
                    print(f"   Mode: {self.performance_mode}")
                    if self.performance_mode == "turbo":
                        print(f"   🔥 Turbo multipliers: batch={self.turbo_batch_multiplier:.2f}x, epochs={self.turbo_epoch_multiplier:.2f}x")
                    print(f"   Batch size: {dynamic_batch_size} (base: {base_batch_size}, multiplier: {dynamic_batch_size/base_batch_size:.2f}x)")
                    print(f"   Epochs: {dynamic_n_epochs} (base: {base_n_epochs}, multiplier: {dynamic_n_epochs/base_n_epochs:.2f}x)")
                    print(f"   Buffer size: {len(self.agent.states)} samples")
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

