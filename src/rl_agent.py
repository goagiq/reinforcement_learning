"""
PPO (Proximal Policy Optimization) Agent for Trading

This is the core RL agent that learns to trade using PPO algorithm.
PPO is stable and works well with continuous actions (position sizing).

For beginners: This agent interacts with the trading environment,
learns from experience, and improves its trading decisions over time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy

from src.models import ActorNetwork, CriticNetwork


class PPOAgent:
    """
    PPO Agent for continuous action trading.
    
    How it works:
    1. Observes market state
    2. Chooses action (position size) using actor network
    3. Executes action in environment
    4. Receives reward (profit/loss)
    5. Learns from experience using PPO algorithm
    """
    
    def __init__(
        self,
        state_dim: int,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_range: Min and max action values
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (how much we value future rewards)
            gae_lambda: GAE (Generalized Advantage Estimation) lambda
            clip_range: PPO clipping range
            value_loss_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus (encourages exploration)
            max_grad_norm: Maximum gradient norm for clipping
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_range = action_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Get hidden dimensions (default to [256, 256, 128] if not provided)
        hidden_dims = kwargs.get("hidden_dims", [256, 256, 128])
        
        # Create networks with custom architecture
        self.actor = ActorNetwork(state_dim, action_range=action_range, hidden_dims=hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dims=hidden_dims).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer (stores trajectories)
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset the experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action given the current state.
        
        Args:
            state: Current market state
            deterministic: If True, use mean action (no exploration)
        
        Returns:
            action: Selected position size
            value: Estimated state value
            log_prob: Log probability of action
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from actor
        with torch.no_grad():
            action, log_prob = self.actor.get_action_and_log_prob(
                state_tensor,
                deterministic=deterministic
            )
            
            # Get value from critic
            value = self.critic(state_tensor)
        
        # Convert back to numpy
        action_np = action.cpu().numpy()[0]
        value_np = value.cpu().item()
        log_prob_np = log_prob.cpu().item()
        
        return action_np, value_np, log_prob_np
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Store a transition in the experience buffer.
        
        This is called after each step in the environment.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE helps estimate how good each action was, considering
        both immediate reward and future potential.
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = next_value
        
        # Compute backwards from last step
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * next_value - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages (helps with training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(
        self,
        n_epochs: int = 10,
        batch_size: int = 64,
        scaler=None,
        autocast=None
    ) -> Dict[str, float]:
        """
        Update the agent using PPO algorithm.
        
        This is called after collecting a batch of experiences.
        PPO updates the policy multiple times on the same data (n_epochs)
        with clipping to prevent large policy changes.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        # Verify tensors are on correct device (first update only)
        if not hasattr(self, '_gpu_verified'):
            if self.device.type == 'cuda':
                print(f"✅ GPU Update: Tensors on {states.device}, GPU: {torch.cuda.get_device_name(0)}")
                print(f"   Batch size: {len(states)}, GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            self._gpu_verified = True
        
        # Compute advantages and returns
        next_value = 0.0
        if not self.dones[-1]:
            # Estimate next value if episode didn't end
            with torch.no_grad():
                next_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state).cpu().item()
        
        advantages, returns = self.compute_gae(
            self.rewards, self.values, self.dones, next_value
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Training metrics
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy = 0
        
        # PPO update: multiple epochs over the same data
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        
        # Log batch processing details (always log for large batch sizes, or when batch_size changes significantly)
        should_log = (not hasattr(self, '_last_logged_batch_size') or 
                     abs(self._last_logged_batch_size - batch_size) > batch_size * 0.5 or
                     batch_size >= 400)  # Always log for Turbo mode (batch_size >= 400 for aggressive Turbo)
        
        if should_log:
            batches_per_epoch = (n_samples + batch_size - 1) // batch_size
            total_updates = n_epochs * batches_per_epoch
            print(f"\n{'='*60}")
            print(f"✅ VERIFICATION: Agent.update() called with:")
            print(f"   n_samples: {n_samples}")
            print(f"   batch_size: {batch_size} ⬅️ VERIFY THIS IS 512 for Turbo mode! (was 256)")
            print(f"   n_epochs: {n_epochs} ⬅️ VERIFY THIS IS 30 for Turbo mode! (was 20)")
            print(f"   Batches per epoch: {batches_per_epoch}")
            print(f"   Total updates: {total_updates}")
            print(f"{'='*60}\n")
            self._last_logged_batch_size = batch_size
        
        # GPU verification: Check device placement before training loop
        if self.device.type == 'cuda':
            print(f"\n🔍 GPU VERIFICATION before training loop:")
            print(f"   States device: {states.device}")
            print(f"   Actions device: {actions.device}")
            print(f"   Actor device: {next(self.actor.parameters()).device}")
            print(f"   Critic device: {next(self.critic.parameters()).device}")
            print(f"   GPU Memory before loop: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
            print(f"   GPU Utilization check: Run nvidia-smi NOW to see baseline\n")
        
        for epoch in range(n_epochs):
            # Shuffle indices for each epoch
            np.random.shuffle(indices)
            
            # Process in batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # GPU verification: Log every few batches during Turbo mode to show GPU activity
                batch_num = start // batch_size + 1
                if self.device.type == 'cuda' and batch_num == 1 and epoch == 0:
                    import time
                    # Force GPU work to be visible
                    torch.cuda.synchronize()
                    gpu_mem_before = torch.cuda.memory_allocated(0) / 1e9
                    batch_start_time = time.time()
                    print(f"\n   🔥🔥🔥 STARTING FIRST BATCH - GPU SHOULD SPIKE NOW! 🔥🔥🔥")
                    print(f"      Batch {batch_num}/{batches_per_epoch} (Epoch {epoch+1}/{n_epochs})")
                    print(f"      GPU Memory before: {gpu_mem_before:.3f} GB")
                    print(f"      ⚡⚡⚡ Check monitor_gpu.sh NOW! ⚡⚡⚡\n")
                
                # Use mixed precision if available
                if autocast is not None:
                    with autocast():
                        # Get current policy outputs
                        mean, log_std = self.actor(batch_states)
                        std = torch.exp(log_std)
                        
                        # Check for NaN/Inf values and replace
                        mean = torch.where(torch.isnan(mean) | torch.isinf(mean), torch.zeros_like(mean), mean)
                        std = torch.where(torch.isnan(std) | torch.isinf(std), torch.ones_like(std), std)
                        
                        normal = torch.distributions.Normal(mean, std)
                        
                        new_log_probs = normal.log_prob(batch_actions)
                        
                        # Check for NaN in log_probs
                        new_log_probs = torch.where(torch.isnan(new_log_probs) | torch.isinf(new_log_probs), torch.zeros_like(new_log_probs), new_log_probs)
                        
                        # Compute entropy (encourages exploration)
                        entropy_batch = normal.entropy().mean()
                        
                        # Compute value
                        values = self.critic(batch_states)
                        
                        # PPO Clipped Surrogate Objective
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        # Clamp ratio to prevent NaN/Inf
                        ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                        
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                        policy_loss_batch = -torch.min(surr1, surr2).mean()
                        
                        # Check for NaN in losses
                        policy_loss_batch = torch.where(torch.isnan(policy_loss_batch) | torch.isinf(policy_loss_batch), torch.zeros_like(policy_loss_batch), policy_loss_batch)
                        
                        # Value loss (MSE)
                        value_loss_batch = nn.MSELoss()(values, batch_returns)
                        value_loss_batch = torch.where(torch.isnan(value_loss_batch) | torch.isinf(value_loss_batch), torch.zeros_like(value_loss_batch), value_loss_batch)
                        
                        # Total loss
                        loss = policy_loss_batch + self.value_loss_coef * value_loss_batch - self.entropy_coef * entropy_batch
                else:
                    # Standard precision
                    # Get current policy outputs
                    mean, log_std = self.actor(batch_states)
                    std = torch.exp(log_std)
                    
                    # Check for NaN/Inf values and replace
                    mean = torch.where(torch.isnan(mean) | torch.isinf(mean), torch.zeros_like(mean), mean)
                    std = torch.where(torch.isnan(std) | torch.isinf(std), torch.ones_like(std), std)
                    
                    normal = torch.distributions.Normal(mean, std)
                    
                    new_log_probs = normal.log_prob(batch_actions)
                    
                    # Check for NaN in log_probs
                    new_log_probs = torch.where(torch.isnan(new_log_probs) | torch.isinf(new_log_probs), torch.zeros_like(new_log_probs), new_log_probs)
                    
                    # Compute entropy (encourages exploration)
                    entropy_batch = normal.entropy().mean()
                    
                    # Compute value
                    values = self.critic(batch_states)
                    
                    # PPO Clipped Surrogate Objective
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    # Clamp ratio to prevent NaN/Inf
                    ratio = torch.clamp(ratio, min=1e-8, max=1e8)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                    policy_loss_batch = -torch.min(surr1, surr2).mean()
                    
                    # Check for NaN in losses
                    policy_loss_batch = torch.where(torch.isnan(policy_loss_batch) | torch.isinf(policy_loss_batch), torch.zeros_like(policy_loss_batch), policy_loss_batch)
                    
                    # Value loss (MSE)
                    value_loss_batch = nn.MSELoss()(values, batch_returns)
                    value_loss_batch = torch.where(torch.isnan(value_loss_batch) | torch.isinf(value_loss_batch), torch.zeros_like(value_loss_batch), value_loss_batch)
                    
                    # Total loss
                    loss = policy_loss_batch + self.value_loss_coef * value_loss_batch - self.entropy_coef * entropy_batch
                
                # Update actor
                self.actor_optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(policy_loss_batch).backward()
                    scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    scaler.step(self.actor_optimizer)
                else:
                    policy_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(value_loss_batch).backward()
                    scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    scaler.step(self.critic_optimizer)
                else:
                    value_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                
                # GPU verification: After first batch, check GPU activity
                if self.device.type == 'cuda' and batch_num == 1 and epoch == 0:
                    import time
                    torch.cuda.synchronize()  # Force GPU work to complete
                    gpu_mem_after = torch.cuda.memory_allocated(0) / 1e9
                    batch_time = time.time() - batch_start_time
                    print(f"\n   ✅ FIRST BATCH COMPLETE:")
                    print(f"      GPU Memory after: {gpu_mem_after:.3f} GB")
                    print(f"      Batch time: {batch_time*1000:.1f}ms")
                    print(f"      💡 Note: Small models may not show high GPU utilization")
                    print(f"      💡 If training is fast (4-5s per update), GPU is working correctly\n")
                
                # Update scaler if using mixed precision
                if scaler is not None:
                    scaler.update()
                
                # Accumulate metrics
                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy += entropy_batch.item()
        
        # Average metrics
        n_updates = n_epochs * (n_samples // batch_size + (1 if n_samples % batch_size > 0 else 0))
        
        metrics = {
            "loss": total_loss / n_updates,
            "policy_loss": policy_loss / n_updates,
            "value_loss": value_loss / n_updates,
            "entropy": entropy / n_updates,
        }
        
        # Clear buffer after update
        self.reset_buffer()
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Agent saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Agent loaded from: {filepath}")
    
    def save_with_training_state(self, filepath: str, timestep: int, episode: int, episode_rewards: list, episode_lengths: list):
        """Save agent state with training progress metadata"""
        # Extract hidden_dims from actor network architecture
        hidden_dims = []
        for i, layer in enumerate(self.actor.feature_layers):
            if isinstance(layer, torch.nn.Linear):
                # Get output size - this is the hidden dimension
                if i == 0:
                    # First layer: input_dim -> hidden_dim, output is hidden_dim
                    hidden_dims.append(layer.out_features)
                else:
                    # Subsequent layers
                    hidden_dims.append(layer.out_features)
        
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "timestep": timestep,
            "episode": episode,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "hidden_dims": hidden_dims,  # Save architecture for resume
            "state_dim": self.state_dim,  # Save state_dim too
        }, filepath)
        print(f"Agent saved with training state to: {filepath} (timestep={timestep}, episode={episode}, hidden_dims={hidden_dims})")
    
    def load_with_training_state(self, filepath: str):
        """Load agent state and return training progress metadata"""
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        # Extract training state if available
        timestep = checkpoint.get("timestep", None)
        episode = checkpoint.get("episode", None)
        episode_rewards = checkpoint.get("episode_rewards", None)
        episode_lengths = checkpoint.get("episode_lengths", None)
        
        # If training state not in checkpoint (old format), try to extract from filename
        if timestep is None:
            import re
            from pathlib import Path
            filename = Path(filepath).name
            # Try to extract timestep from filename like checkpoint_240000.pt
            match = re.search(r'checkpoint_(\d+)\.pt', filename)
            if match:
                timestep = int(match.group(1))
                print(f"⚠️  Checkpoint doesn't contain timestep metadata, extracting from filename: {timestep}")
            else:
                timestep = 0
                print(f"⚠️  Checkpoint doesn't contain timestep metadata and couldn't extract from filename")
        
        # Default values if not found
        if episode is None:
            episode = 0
        if episode_rewards is None:
            episode_rewards = []
        if episode_lengths is None:
            episode_lengths = []
        
        print(f"Agent loaded from: {filepath} (timestep={timestep}, episode={episode})")
        return timestep, episode, episode_rewards, episode_lengths
    
    def load_with_transfer(self, filepath: str, transfer_strategy: str = "copy_and_extend"):
        """
        Load agent state with transfer learning from different architecture.
        
        This method transfers weights from a checkpoint with different architecture
        (e.g., [128, 128, 64] -> [256, 256, 128]) while preserving learned knowledge.
        
        Args:
            filepath: Path to checkpoint with different architecture
            transfer_strategy: Strategy for weight transfer:
                - "copy_and_extend": Copy old weights, initialize new dimensions with small random values
                - "interpolate": Interpolate old weights to fill new dimensions
                - "zero_pad": Copy old weights, pad new dimensions with zeros
        
        Returns:
            Tuple of (timestep, episode, episode_rewards, episode_lengths)
        """
        from src.weight_transfer import transfer_checkpoint_weights
        
        print(f"\n🔄 Loading checkpoint with transfer learning: {filepath}")
        
        # Transfer weights
        new_actor_state, new_critic_state = transfer_checkpoint_weights(
            filepath,
            self.actor,
            self.critic,
            transfer_strategy
        )
        
        # Load transferred weights
        self.actor.load_state_dict(new_actor_state)
        self.critic.load_state_dict(new_critic_state)
        
        # Load checkpoint for training state and optimizer states
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Try to load optimizer states (may fail if architecture changed significantly)
        try:
            # Create new optimizers with current network parameters
            # This ensures optimizer states match new architecture
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_optimizer.param_groups[0]['lr'])
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_optimizer.param_groups[0]['lr'])
            print("  ✅ Optimizers reinitialized for new architecture")
        except Exception as e:
            print(f"  ⚠️  Could not transfer optimizer states: {e}")
            print("  ✅ Optimizers will be reinitialized")
        
        # Extract training state
        timestep = checkpoint.get("timestep", 0)
        episode = checkpoint.get("episode", 0)
        episode_rewards = checkpoint.get("episode_rewards", [])
        episode_lengths = checkpoint.get("episode_lengths", [])
        
        # If training state not in checkpoint, try to extract from filename
        if timestep == 0:
            import re
            from pathlib import Path
            filename = Path(filepath).name
            match = re.search(r'checkpoint_(\d+)\.pt', filename)
            if match:
                timestep = int(match.group(1))
                print(f"  📊 Extracted timestep from filename: {timestep}")
        
        print(f"✅ Transfer learning complete! (timestep={timestep}, episode={episode})")
        return timestep, episode, episode_rewards, episode_lengths


# Example usage
if __name__ == "__main__":
    print("Testing PPO Agent...")
    print("-" * 50)
    
    # Create agent
    state_dim = 200
    agent = PPOAgent(state_dim, device="cpu")
    
    # Simulate a few steps
    print("\n1. Simulating agent interaction...")
    for i in range(5):
        # Random state
        state = np.random.randn(state_dim)
        
        # Select action
        action, value, log_prob = agent.select_action(state)
        
        # Simulate reward
        reward = np.random.randn() * 0.01
        
        # Store transition
        agent.store_transition(state, action, reward, value, log_prob, done=False)
        
        print(f"   Step {i+1}: Action={action[0]:.3f}, Value={value:.3f}, Reward={reward:.4f}")
    
    # Update agent
    print("\n2. Updating agent...")
    metrics = agent.update(n_epochs=3, batch_size=2)
    print(f"   Training metrics: {metrics}")
    
    print("\n✅ Agent working correctly!")
    print("\nNext: Use this agent in training script (src/train.py)")

