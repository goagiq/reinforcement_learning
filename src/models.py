"""
Neural Network Models for RL Trading Agent

Implements Actor-Critic architecture:
- Actor: Policy network that outputs position sizing decisions
- Critic: Value network that estimates state values

For beginners: This is the "brain" of our trading agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy Network)
    
    Takes market state as input and outputs:
    - Mean position size (-1.0 to 1.0)
    - Standard deviation (for exploration)
    
    This network learns the optimal trading policy.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256, 128],
        action_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Initialize Actor Network.
        
        Args:
            state_dim: Size of state space (market features)
            hidden_dims: Hidden layer sizes
            action_range: Min and max action values (position size range)
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_range = action_range
        self.action_min, self.action_max = action_range
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Prevent overfitting
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output heads
        # Mean: The recommended position size
        self.mean_head = nn.Linear(input_dim, 1)
        
        # Log standard deviation: For exploration (we use log to ensure it's positive)
        self.log_std_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Initialize mean head to output near zero (neutral position)
        nn.init.constant_(self.mean_head.bias, 0.0)
        # Initialize log_std to small value (moderate exploration)
        nn.init.constant_(self.log_std_head.bias, -1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Market state features [batch_size, state_dim]
        
        Returns:
            mean: Mean position size [batch_size, 1]
            log_std: Log standard deviation [batch_size, 1]
        """
        # Extract features
        features = self.feature_layers(state)
        
        # Get mean position size
        mean = self.mean_head(features)
        # Clip to action range
        mean = torch.tanh(mean) * 0.5 * (self.action_max - self.action_min) + 0.5 * (self.action_max + self.action_min)
        
        # Check for NaN values and replace with 0
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        
        # Get log standard deviation (clamped for stability)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # Check for NaN values in log_std and replace with default
        log_std = torch.where(torch.isnan(log_std), torch.full_like(log_std, -1.0), log_std)
        
        return mean, log_std
    
    def get_action_and_log_prob(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            state: Market state
            deterministic: If True, return mean (no exploration)
        
        Returns:
            action: Sampled position size
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            # Return mean (no exploration)
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            # Sample from normal distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            
            # Clip to action range
            action = torch.clamp(action, self.action_min, self.action_max)
            
            # Calculate log probability
            log_prob = normal.log_prob(action)
            # Adjust for tanh squashing if needed (in this case, we already clipped)
            
        return action, log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability of a given action.
        
        Args:
            state: Market state
            action: Action to evaluate
        
        Returns:
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action)
        
        return log_prob


class CriticNetwork(nn.Module):
    """
    Critic Network (Value Network)
    
    Estimates the value (expected future return) of being in a given state.
    This helps the actor learn better policies.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256, 128]
    ):
        """
        Initialize Critic Network.
        
        Args:
            state_dim: Size of state space
            hidden_dims: Hidden layer sizes
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output: Single value (state value)
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Initialize value head bias to zero
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Market state features [batch_size, state_dim]
        
        Returns:
            value: Estimated state value [batch_size, 1]
        """
        features = self.feature_layers(state)
        value = self.value_head(features)
        return value


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic Network
    
    Shares feature extraction between actor and critic for efficiency.
    This is a common pattern in RL.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list = [256, 256, 128],
        action_range: Tuple[float, float] = (-1.0, 1.0),
        shared_layers: int = 2
    ):
        """
        Initialize combined Actor-Critic.
        
        Args:
            state_dim: Size of state space
            hidden_dims: Hidden layer sizes
            action_range: Action value range
            shared_layers: Number of shared layers before splitting
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_range = action_range
        
        # Shared feature extraction
        shared_layers_list = []
        input_dim = state_dim
        
        for i in range(min(shared_layers, len(hidden_dims))):
            shared_layers_list.extend([
                nn.Linear(input_dim, hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dims[i]
        
        self.shared_layers = nn.Sequential(*shared_layers_list)
        
        # Actor head (policy)
        actor_hidden = hidden_dims[shared_layers:] if shared_layers < len(hidden_dims) else []
        if not actor_hidden:
            actor_hidden = [128]  # Default if no remaining layers
        
        actor_layers = []
        for dim in actor_hidden:
            actor_layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = dim
        
        self.actor_feature_layers = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()
        
        self.actor_mean = nn.Linear(input_dim, 1)
        self.actor_log_std = nn.Linear(input_dim, 1)
        
        # Critic head (value)
        critic_hidden = hidden_dims[shared_layers:] if shared_layers < len(hidden_dims) else []
        if not critic_hidden:
            critic_hidden = [128]
        
        critic_layers = []
        input_dim = hidden_dims[min(shared_layers, len(hidden_dims) - 1)] if shared_layers > 0 else state_dim
        for dim in critic_hidden:
            critic_layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = dim
        
        self.critic_feature_layers = nn.Sequential(*critic_layers) if critic_layers else nn.Identity()
        self.critic_value = nn.Linear(input_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.constant_(self.actor_log_std.bias, -1.0)
        nn.init.constant_(self.critic_value.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Actor mean action
            log_std: Actor log std
            value: Critic value
        """
        # Shared features
        shared_features = self.shared_layers(state)
        
        # Actor
        actor_features = self.actor_feature_layers(shared_features)
        mean = torch.tanh(self.actor_mean(actor_features)) * 0.5
        log_std = torch.clamp(self.actor_log_std(actor_features), min=-20, max=2)
        
        # Critic
        critic_features = self.critic_feature_layers(shared_features)
        value = self.critic_value(critic_features)
        
        return mean, log_std, value


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Network Models...")
    print("-" * 50)
    
    # Test parameters
    state_dim = 200  # Example state dimension
    batch_size = 32
    
    # Create networks
    print("\n1. Testing separate Actor and Critic...")
    actor = ActorNetwork(state_dim)
    critic = CriticNetwork(state_dim)
    
    # Test forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    
    mean, log_std = actor(dummy_state)
    value = critic(dummy_state)
    
    print(f"   Actor output - Mean shape: {mean.shape}, Log std shape: {log_std.shape}")
    print(f"   Critic output - Value shape: {value.shape}")
    
    # Test action sampling
    action, log_prob = actor.get_action_and_log_prob(dummy_state)
    print(f"   Sampled action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    print(f"   Action range: [{action.min():.2f}, {action.max():.2f}]")
    
    # Test combined network
    print("\n2. Testing combined Actor-Critic...")
    ac_network = ActorCriticNetwork(state_dim)
    mean, log_std, value = ac_network(dummy_state)
    print(f"   Mean shape: {mean.shape}, Log std shape: {log_std.shape}, Value shape: {value.shape}")
    
    print("\n✅ All networks working correctly!")
    print("\nNext steps:")
    print("1. These networks will be used in the PPO agent")
    print("2. The actor learns the trading policy")
    print("3. The critic helps estimate value and guide learning")

