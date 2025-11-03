"""
Analyze saved model architecture and extract training information
"""

import torch
from pathlib import Path

def analyze_checkpoint(model_path: str):
    """Analyze a model checkpoint"""
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {Path(model_path).name}")
    print('='*60)
    
    # Check what's in the checkpoint
    print("\n📦 Checkpoint Contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            size = sum(p.numel() if hasattr(p, 'numel') else 0 for p in checkpoint[key].values() if torch.is_tensor(p))
            print(f"   {key}: {len(checkpoint[key])} tensors, ~{size/1e6:.2f}M params")
        else:
            print(f"   {key}: {type(checkpoint[key])}")
    
    # Analyze actor network structure
    if 'actor_state_dict' in checkpoint:
        actor = checkpoint['actor_state_dict']
        
        print("\n🧠 Actor Network Architecture:")
        
        # Find input dimension
        if 'feature_layers.0.weight' in actor:
            input_dim = actor['feature_layers.0.weight'].shape[1]
            first_layer_size = actor['feature_layers.0.weight'].shape[0]
            print(f"   Input dimension: {input_dim}")
            print(f"   First hidden layer: {first_layer_size}")
        
        # Find all layer sizes
        layer_sizes = []
        i = 0
        while f'feature_layers.{i}.weight' in actor:
            weight = actor[f'feature_layers.{i}.weight']
            layer_sizes.append(f"{weight.shape[0]}x{weight.shape[1]}")
            i += 3  # Skip ReLU and Dropout
        
        if layer_sizes:
            print(f"   Hidden layers: {' → '.join(layer_sizes)}")
        
        # Output heads
        if 'mean_head.weight' in actor:
            output_size = actor['mean_head.weight'].shape[0]
            print(f"   Output size: {output_size}")
    
    # Analyze critic network
    if 'critic_state_dict' in checkpoint:
        critic = checkpoint['critic_state_dict']
        
        print("\n💎 Critic Network Architecture:")
        
        if 'feature_layers.0.weight' in critic:
            input_dim = critic['feature_layers.0.weight'].shape[1]
            first_layer_size = critic['feature_layers.0.weight'].shape[0]
            print(f"   Input dimension: {input_dim}")
            print(f"   First hidden layer: {first_layer_size}")
    
    return checkpoint

if __name__ == "__main__":
    # Analyze best model
    if Path("models/best_model.pt").exists():
        analyze_checkpoint("models/best_model.pt")
    
    # Analyze latest checkpoint
    latest = sorted(Path("models").glob("checkpoint_*.pt"), 
                   key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0,
                   reverse=True)
    if latest:
        analyze_checkpoint(str(latest[0]))

