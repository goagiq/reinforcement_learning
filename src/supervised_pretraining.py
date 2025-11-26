"""
Supervised Pre-training for RL Trading Agent

Pre-trains the actor network on historical trading data before RL fine-tuning.
This helps the agent learn basic market patterns and reduces random exploration.

For quant traders: This is like teaching the agent "market basics" before letting
it learn from experience. Similar to how human traders study charts before trading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.trading_hours import TradingHoursManager
from src.utils.colors import success, info, warn, error


class SupervisedPretrainer:
    """
    Pre-trains the actor network using supervised learning on historical data.
    
    Strategy:
    1. Load historical OHLCV data
    2. Generate optimal action labels based on future returns
    3. Train actor network to predict optimal actions
    4. Save pre-trained weights for RL fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        data_extractor: DataExtractor,
        device: str = "cpu"
    ):
        """
        Initialize supervised pre-trainer.
        
        Args:
            config: Training configuration dictionary
            data_extractor: DataExtractor instance for loading historical data
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.data_extractor = data_extractor
        self.device = torch.device(device)
        
        # Pre-training configuration
        pretrain_config = config.get("pretraining", {})
        self.enabled = pretrain_config.get("enabled", False)
        self.lookahead_bars = pretrain_config.get("lookahead_bars", 20)  # Look ahead 20 bars
        self.return_threshold = pretrain_config.get("return_threshold", 0.02)  # 2% return threshold
        self.batch_size = pretrain_config.get("batch_size", 256)
        self.epochs = pretrain_config.get("epochs", 10)
        self.learning_rate = pretrain_config.get("learning_rate", 1e-3)
        self.validation_split = pretrain_config.get("validation_split", 0.2)
        
        # Labeling strategy
        self.labeling_strategy = pretrain_config.get("labeling_strategy", "simple_return")
        # Options: "simple_return", "sharpe_based", "volatility_adjusted"
    
    def generate_labels(
        self,
        data: pd.DataFrame,
        lookahead: int,
        return_threshold: float
    ) -> np.ndarray:
        """
        Generate optimal action labels from historical data.
        
        Strategy: Look ahead N bars, calculate future return.
        - If return > threshold: label = +1.0 (buy)
        - If return < -threshold: label = -1.0 (sell)
        - Otherwise: label = 0.0 (hold)
        
        Args:
            data: DataFrame with OHLCV data
            lookahead: Number of bars to look ahead
            return_threshold: Minimum return to trigger buy/sell
        
        Returns:
            Array of optimal actions (same length as data)
        """
        if len(data) < lookahead + 1:
            # Not enough data for lookahead
            return np.zeros(len(data))
        
        labels = np.zeros(len(data))
        prices = data['close'].values
        
        for i in range(len(data) - lookahead):
            current_price = prices[i]
            future_price = prices[i + lookahead]
            
            # Calculate future return
            future_return = (future_price - current_price) / current_price
            
            # Generate label based on return
            if future_return > return_threshold:
                labels[i] = 1.0  # Buy signal
            elif future_return < -return_threshold:
                labels[i] = -1.0  # Sell signal
            else:
                labels[i] = 0.0  # Hold signal
        
        # Last lookahead bars: use hold (can't look ahead)
        labels[-lookahead:] = 0.0
        
        return labels
    
    def prepare_training_data(
        self,
        env: TradingEnvironment
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data: states and optimal action labels.
        
        Args:
            env: TradingEnvironment instance (for state extraction)
        
        Returns:
            Tuple of (states, labels) as tensors
        """
        print(info("\n[PRETRAIN] Preparing training data from historical data..."))
        
        # Get data from environment
        primary_data = env.data[min(env.timeframes)]
        
        # Generate labels
        labels = self.generate_labels(
            primary_data,
            self.lookahead_bars,
            self.return_threshold
        )
        
        # Extract states for each data point
        states = []
        valid_labels = []
        
        # Need at least lookahead_bars + some history for indicators
        min_history = max(env.lookback_bars, self.lookahead_bars + 10)
        
        for i in range(min_history, len(primary_data) - self.lookahead_bars):
            try:
                # Extract state at this point using _get_state_features
                # Temporarily set current_step to extract state
                original_step = env.current_step
                env.current_step = i
                state = env._get_state_features(i)
                env.current_step = original_step
                
                if state is not None and not np.isnan(state).any() and len(state) > 0:
                    states.append(state)
                    valid_labels.append(labels[i])
            except Exception as e:
                # Skip if state extraction fails
                continue
        
        if len(states) == 0:
            error_msg = "No valid states extracted from historical data"
            print(error(f"[PRETRAIN] {error_msg}"))
            raise ValueError(error_msg)
        
        states_array = np.array(states)
        labels_array = np.array(valid_labels)
        
        print(success(f"[PRETRAIN] Prepared {len(states)} training samples"))
        print(info(f"  Label distribution: Buy={np.sum(labels_array > 0)}, Sell={np.sum(labels_array < 0)}, Hold={np.sum(labels_array == 0)}"))
        
        return torch.FloatTensor(states_array), torch.FloatTensor(labels_array)
    
    def pretrain_actor(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict:
        """
        Pre-train actor network using supervised learning.
        
        Args:
            actor: Actor network to pre-train
            labels: Optimal action labels
        
        Returns:
            Dictionary with training metrics
        """
        print(info(f"\n[PRETRAIN] Starting supervised pre-training..."))
        print(info(f"  Epochs: {self.epochs}"))
        print(info(f"  Batch size: {self.batch_size}"))
        print(info(f"  Learning rate: {self.learning_rate}"))
        
        # Split into train/validation
        n_train = int(len(states) * (1 - self.validation_split))
        train_states = states[:n_train]
        train_labels = labels[:n_train]
        val_states = states[n_train:]
        val_labels = labels[n_train:]
        
        # Optimizer
        optimizer = optim.Adam(actor.parameters(), lr=self.learning_rate)
        
        # Loss function: MSE for continuous actions
        criterion = nn.MSELoss()
        
        # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        actor.train()
        
        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            n_batches = 0
            
            # Shuffle training data
            indices = torch.randperm(len(train_states))
            train_states_shuffled = train_states[indices]
            train_labels_shuffled = train_labels[indices]
            
            for i in range(0, len(train_states_shuffled), self.batch_size):
                batch_states = train_states_shuffled[i:i+self.batch_size].to(self.device)
                batch_labels = train_labels_shuffled[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass: get action mean (we'll use mean as prediction)
                action_dist = actor(batch_states)
                if isinstance(action_dist, tuple):
                    action_mean, _ = action_dist
                else:
                    action_mean = action_dist
                
                # Predict action (mean of distribution)
                predicted_actions = action_mean.squeeze()
                
                # Loss: difference between predicted and optimal action
                loss = criterion(predicted_actions, batch_labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0.0
            train_losses.append(avg_train_loss)
            
            # Validation
            actor.eval()
            with torch.no_grad():
                val_action_dist = actor(val_states.to(self.device))
                if isinstance(val_action_dist, tuple):
                    val_action_mean, _ = val_action_dist
                else:
                    val_action_mean = val_action_dist
                val_predicted = val_action_mean.squeeze()
                val_loss = criterion(val_predicted, val_labels.to(self.device))
                avg_val_loss = val_loss.item()
                val_losses.append(avg_val_loss)
            
            actor.train()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(info(f"[PRETRAIN] Early stopping at epoch {epoch+1}"))
                    break
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(info(f"  Epoch {epoch+1}/{self.epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"))
        
        print(success(f"[PRETRAIN] Pre-training complete! Best validation loss: {best_val_loss:.6f}"))
        
        # Calculate final training metrics
        final_train_loss = train_losses[-1] if train_losses else 0.0
        final_val_loss = val_losses[-1] if val_losses else 0.0
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "epochs_trained": epoch + 1,
            "n_samples": len(train_states),
            "n_validation_samples": len(val_states)
        }
    
    def run_pretraining(
        self,
        actor: nn.Module,
        env: TradingEnvironment
    ) -> Dict:
        """
        Run complete pre-training pipeline.
        
        Args:
            actor: Actor network to pre-train
            env: TradingEnvironment for state extraction
        
        Returns:
            Dictionary with training metrics and results
        """
        if not self.enabled:
            print(warn("[PRETRAIN] Pre-training is disabled in config"))
            return {}
        
        print(info("\n" + "="*70))
        print(info("SUPERVISED PRE-TRAINING"))
        print(info("="*70))
        
        try:
            # Prepare training data
            states, labels = self.prepare_training_data(env)
            
            # Pre-train actor
            metrics = self.pretrain_actor(actor, states, labels)
            
            print(success("\n[PRETRAIN] Pre-training completed successfully!"))
            print(info("="*70 + "\n"))
            
            return metrics
            
        except Exception as e:
            print(error(f"[PRETRAIN] Pre-training failed: {e}"))
            import traceback
            traceback.print_exc()
            return {}

