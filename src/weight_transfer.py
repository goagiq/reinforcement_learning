"""
Weight Transfer Utility for Transfer Learning

Transfers weights from a smaller architecture to a larger one while preserving
learned knowledge. Supports expanding hidden layer dimensions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy


def transfer_linear_weights(
    old_weight: torch.Tensor,
    old_bias: Optional[torch.Tensor],
    new_layer: nn.Linear,
    strategy: str = "copy_and_extend"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Transfer weights from old linear layer to new (potentially larger) layer.
    
    Args:
        old_weight: Old layer weight tensor [out_dim, in_dim]
        old_bias: Old layer bias tensor [out_dim] or None
        new_layer: New layer to transfer weights to
        strategy: Transfer strategy:
            - "copy_and_extend": Copy old weights, initialize new dimensions with small random values
            - "interpolate": Interpolate old weights to fill new dimensions
            - "zero_pad": Copy old weights, pad new dimensions with zeros
    
    Returns:
        Tuple of (new_weight, new_bias) tensors
    """
    new_out_dim, new_in_dim = new_layer.weight.shape
    old_out_dim, old_in_dim = old_weight.shape
    
    # Initialize new weight with Xavier uniform (will be overwritten where applicable)
    new_weight = torch.empty_like(new_layer.weight)
    nn.init.xavier_uniform_(new_weight)
    
    # Copy compatible weights
    copy_out_dim = min(old_out_dim, new_out_dim)
    copy_in_dim = min(old_in_dim, new_in_dim)
    
    new_weight[:copy_out_dim, :copy_in_dim] = old_weight[:copy_out_dim, :copy_in_dim]
    
    # Handle extended dimensions
    if new_out_dim > old_out_dim:
        if strategy == "copy_and_extend":
            # Initialize new output neurons with small random values (10% of original scale)
            scale = old_weight.std().item() * 0.1
            new_weight[old_out_dim:, :copy_in_dim] = torch.randn(
                new_out_dim - old_out_dim, copy_in_dim
            ) * scale
        elif strategy == "interpolate":
            # Interpolate from existing neurons (average of first few)
            avg_weight = old_weight[:min(3, old_out_dim)].mean(dim=0)
            new_weight[old_out_dim:, :copy_in_dim] = avg_weight.unsqueeze(0).expand(
                new_out_dim - old_out_dim, -1
            )
        elif strategy == "zero_pad":
            new_weight[old_out_dim:, :copy_in_dim] = 0.0
    
    if new_in_dim > old_in_dim:
        # For extended input dimensions, pad with small random values
        scale = old_weight.std().item() * 0.1
        new_weight[:, old_in_dim:] = torch.randn(
            new_out_dim, new_in_dim - old_in_dim
        ) * scale
    
    # Handle bias
    new_bias = None
    if old_bias is not None and new_layer.bias is not None:
        new_bias = torch.empty_like(new_layer.bias)
        new_bias[:copy_out_dim] = old_bias[:copy_out_dim]
        
        if new_out_dim > old_out_dim:
            if strategy == "copy_and_extend":
                new_bias[old_out_dim:] = 0.0  # Initialize new biases to zero
            elif strategy == "interpolate":
                avg_bias = old_bias[:min(3, old_out_dim)].mean()
                new_bias[old_out_dim:] = avg_bias
            elif strategy == "zero_pad":
                new_bias[old_out_dim:] = 0.0
    elif new_layer.bias is not None:
        nn.init.constant_(new_bias, 0.0)
    
    return new_weight, new_bias


def transfer_network_weights(
    old_state_dict: Dict[str, torch.Tensor],
    new_network: nn.Module,
    old_hidden_dims: List[int],
    new_hidden_dims: List[int],
    old_state_dim: int,  # Old state dimension (e.g., 900)
    transfer_strategy: str = "copy_and_extend"
) -> Dict[str, torch.Tensor]:
    """
    Transfer weights from old network architecture to new network architecture.
    
    Args:
        old_state_dict: State dict from old network
        new_network: New network module (ActorNetwork or CriticNetwork)
        old_hidden_dims: Hidden dimensions of old network [128, 128, 64]
        new_hidden_dims: Hidden dimensions of new network [256, 256, 128]
        old_state_dim: Old state dimension (e.g., 900)
        transfer_strategy: Strategy for transferring weights
    
    Returns:
        New state dict with transferred weights
    
    Note:
        This function handles state_dim increases. If new_state_dim > old_state_dim,
        the new input dimensions will be initialized with small random values.
    """
    new_state_dict = {}
    
    # Get actual new state_dim from first layer
    first_layer = dict(new_network.named_modules())["feature_layers.0"]
    new_state_dim = first_layer.weight.shape[1]
    
    # Log state_dim change if applicable
    if old_state_dim != new_state_dim:
        print(f"  📊 State dimension change: {old_state_dim} → {new_state_dim} (+{new_state_dim - old_state_dim})")
    
    # Calculate layer indices
    # Structure: feature_layers.0 (Linear), .1 (ReLU), .2 (Dropout), .3 (Linear), ...
    old_layer_dims = [old_state_dim] + old_hidden_dims
    new_layer_dims = [new_state_dim] + new_hidden_dims
    
    # Transfer feature layers
    layer_idx = 0
    num_layers_to_transfer = min(len(old_hidden_dims), len(new_hidden_dims))
    
    for i in range(len(new_hidden_dims)):
        # Linear layer
        old_linear_key = f"feature_layers.{layer_idx}.weight"
        old_bias_key = f"feature_layers.{layer_idx}.bias"
        new_linear_key = f"feature_layers.{layer_idx}.weight"
        new_bias_key = f"feature_layers.{layer_idx}.bias"
        
        # Get new layer
        new_layer = dict(new_network.named_modules())[
            f"feature_layers.{layer_idx}"
        ]
        
        if i < num_layers_to_transfer and old_linear_key in old_state_dict:
            # Transfer from old layer
            old_weight = old_state_dict[old_linear_key]
            old_bias = old_state_dict.get(old_bias_key)
            
            new_weight, new_bias = transfer_linear_weights(
                old_weight, old_bias, new_layer, transfer_strategy
            )
            
            new_state_dict[new_linear_key] = new_weight
            if new_bias is not None:
                new_state_dict[new_bias_key] = new_bias
            
            old_dim_str = f"{old_layer_dims[i]} -> {old_layer_dims[i+1]}" if i < len(old_hidden_dims) else "N/A"
            print(f"  ✅ Transferred layer {i+1}: {old_dim_str} → {new_layer_dims[i]} -> {new_layer_dims[i+1]}")
        else:
            # Initialize new layer (either not in old network or new layer beyond old network)
            new_state_dict[new_linear_key] = new_layer.weight.clone()
            if new_layer.bias is not None:
                new_state_dict[new_bias_key] = new_layer.bias.clone()
            print(f"  🆕 Initialized new layer {i+1}: {new_layer_dims[i]} -> {new_layer_dims[i+1]}")
        
        layer_idx += 3  # Skip ReLU and Dropout
    
    # Transfer output head(s)
    # For Actor: mean_head and log_std_head
    # For Critic: value_head
    
    # Check if it's Actor or Critic by looking for mean_head
    if "mean_head.weight" in old_state_dict:
        # Actor network
        for head_name in ["mean_head", "log_std_head"]:
            old_weight_key = f"{head_name}.weight"
            old_bias_key = f"{head_name}.bias"
            
            if old_weight_key in old_state_dict:
                old_weight = old_state_dict[old_weight_key]  # [1, last_hidden_dim]
                old_bias = old_state_dict.get(old_bias_key)
                
                new_layer = dict(new_network.named_modules())[head_name]
                new_weight, new_bias = transfer_linear_weights(
                    old_weight, old_bias, new_layer, transfer_strategy
                )
                
                new_state_dict[old_weight_key] = new_weight
                if new_bias is not None:
                    new_state_dict[old_bias_key] = new_bias
                
                print(f"  ✅ Transferred {head_name}: {old_hidden_dims[-1]} -> 1 → {new_hidden_dims[-1]} -> 1")
    
    elif "value_head.weight" in old_state_dict:
        # Critic network
        old_weight = old_state_dict["value_head.weight"]
        old_bias = old_state_dict.get("value_head.bias")
        
        new_layer = dict(new_network.named_modules())["value_head"]
        new_weight, new_bias = transfer_linear_weights(
            old_weight, old_bias, new_layer, transfer_strategy
        )
        
        new_state_dict["value_head.weight"] = new_weight
        if new_bias is not None:
            new_state_dict["value_head.bias"] = new_bias
        
        print(f"  ✅ Transferred value_head: {old_hidden_dims[-1]} -> 1 → {new_hidden_dims[-1]} -> 1")
    
    return new_state_dict


def transfer_checkpoint_weights(
    checkpoint_path: str,
    new_actor: nn.Module,
    new_critic: nn.Module,
    transfer_strategy: str = "copy_and_extend"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Transfer weights from checkpoint to new actor and critic networks.
    
    Args:
        checkpoint_path: Path to old checkpoint
        new_actor: New ActorNetwork instance
        new_critic: New CriticNetwork instance
        transfer_strategy: Transfer strategy
    
    Returns:
        Tuple of (new_actor_state_dict, new_critic_state_dict)
    """
    print(f"\n🔄 Transferring weights from: {checkpoint_path}")
    print(f"   Strategy: {transfer_strategy}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    old_actor_state = checkpoint.get("actor_state_dict", {})
    old_critic_state = checkpoint.get("critic_state_dict", {})
    
    # Extract architecture info
    old_hidden_dims = checkpoint.get("hidden_dims", [128, 128, 64])
    old_state_dim = checkpoint.get("state_dim", 900)
    
    # Infer new architecture from networks
    # Get first layer to determine state_dim
    first_layer = dict(new_actor.named_modules())["feature_layers.0"]
    new_state_dim = first_layer.weight.shape[1]
    
    # Infer hidden_dims from network structure
    new_hidden_dims = []
    layer_idx = 0
    while f"feature_layers.{layer_idx}.weight" in dict(new_actor.named_parameters()):
        layer = dict(new_actor.named_modules())[f"feature_layers.{layer_idx}"]
        new_hidden_dims.append(layer.weight.shape[0])
        layer_idx += 3
    
    print(f"\n📐 Architecture Mapping:")
    print(f"   Old: state_dim={old_state_dim}, hidden_dims={old_hidden_dims}")
    print(f"   New: state_dim={new_state_dim}, hidden_dims={new_hidden_dims}")
    
    # Handle state dimension changes
    if old_state_dim != new_state_dim:
        if new_state_dim < old_state_dim:
            raise ValueError(
                f"State dimension cannot decrease: old={old_state_dim}, new={new_state_dim}. "
                f"Cannot transfer weights when input dimension decreases. "
                f"Use old_state_dim={new_state_dim} or retrain from scratch."
            )
        else:
            # State dimension increased - allow transfer
            print(f"\n⚠️  State dimension increased: {old_state_dim} → {new_state_dim}")
            print(f"   New input dimensions (+{new_state_dim - old_state_dim}) will be initialized with small random values")
            print(f"   Existing weights will be preserved")
    
    # Transfer actor weights
    # Note: transfer_network_weights will detect new_state_dim from the network
    # and handle input dimension changes automatically via transfer_linear_weights
    print(f"\n🧠 Transferring Actor Network:")
    new_actor_state = transfer_network_weights(
        old_actor_state,
        new_actor,
        old_hidden_dims,
        new_hidden_dims,
        old_state_dim,  # Pass old_state_dim, function will detect new_state_dim from network
        transfer_strategy
    )
    
    # Transfer critic weights
    print(f"\n💎 Transferring Critic Network:")
    new_critic_state = transfer_network_weights(
        old_critic_state,
        new_critic,
        old_hidden_dims,
        new_hidden_dims,
        old_state_dim,  # Pass old_state_dim, function will detect new_state_dim from network
        transfer_strategy
    )
    
    print(f"\n✅ Weight transfer complete!")
    print(f"   Transferred {len(new_actor_state)} actor parameters")
    print(f"   Transferred {len(new_critic_state)} critic parameters")
    
    return new_actor_state, new_critic_state


if __name__ == "__main__":
    # Test weight transfer
    from src.models import ActorNetwork, CriticNetwork
    
    # Create old architecture
    old_actor = ActorNetwork(state_dim=900, hidden_dims=[128, 128, 64])
    old_critic = CriticNetwork(state_dim=900, hidden_dims=[128, 128, 64])
    
    # Create new architecture
    new_actor = ActorNetwork(state_dim=900, hidden_dims=[256, 256, 128])
    new_critic = CriticNetwork(state_dim=900, hidden_dims=[256, 256, 128])
    
    # Save old model as test checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        test_checkpoint = f.name
        torch.save({
            "actor_state_dict": old_actor.state_dict(),
            "critic_state_dict": old_critic.state_dict(),
            "hidden_dims": [128, 128, 64],
            "state_dim": 900
        }, test_checkpoint)
    
    # Transfer weights
    new_actor_state, new_critic_state = transfer_checkpoint_weights(
        test_checkpoint,
        new_actor,
        new_critic,
        transfer_strategy="copy_and_extend"
    )
    
    # Load transferred weights
    new_actor.load_state_dict(new_actor_state)
    new_critic.load_state_dict(new_critic_state)
    
    print("\n✅ Test completed successfully!")

