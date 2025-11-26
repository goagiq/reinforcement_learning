"""
Check checkpoint compatibility with current forecast caching fix
"""
import torch
from pathlib import Path
import json

def check_checkpoint(checkpoint_path: str):
    """Check if checkpoint is compatible with fixed forecast caching"""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
        
        print(f"\n{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
        
        # Extract info
        timestep = checkpoint.get('timestep', 'Unknown')
        episode = checkpoint.get('episode', 'Unknown')
        state_dim = None
        
        # Try to get state_dim from model
        if 'actor_state_dict' in checkpoint:
            actor_state = checkpoint['actor_state_dict']
            # Find first linear layer input size
            for key, value in actor_state.items():
                if 'weight' in key and len(value.shape) == 2:
                    state_dim = value.shape[1]
                    break
        
        print(f"Timestep: {timestep:,}")
        print(f"Episode: {episode}")
        if state_dim:
            print(f"State Dimension: {state_dim}")
        
        # Check if state_dim matches current config
        try:
            import yaml
            with open('configs/train_config_adaptive.yaml', 'r') as f:
                config = yaml.safe_load(f)
            expected_state_dim = config.get('environment', {}).get('state_features', 908)
            
            if state_dim:
                if state_dim == expected_state_dim:
                    print(f"[OK] State dimension matches config ({expected_state_dim})")
                else:
                    print(f"[WARN] State dimension mismatch: checkpoint={state_dim}, config={expected_state_dim}")
                    print(f"   This checkpoint may need transfer learning")
        except Exception as e:
            print(f"[WARN] Could not verify config: {e}")
        
        # Recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        print(f"{'='*80}")
        
        if timestep == 'Unknown' or isinstance(timestep, str):
            print("[WARN] Cannot determine checkpoint age")
            print("   Recommendation: Start fresh or use latest checkpoint")
        else:
            # Forecast caching was just added and fixed
            # If checkpoint is very recent (last few hours/days), it might have buggy caching
            print(f"Checkpoint Age: {timestep:,} timesteps")
            
            # Check state dimension to determine if forecast features were used
            if state_dim == 900:
                print("[OK] SAFE: This checkpoint has state_dim=900 (no forecast features)")
                print("   Trained BEFORE forecast caching was added")
                print("   [RECOMMENDED] Use this checkpoint - will need transfer learning for forecast features")
            elif state_dim == 908:
                if timestep < 2000000:
                    print("[WARN] CAUTION: This checkpoint has state_dim=908 (with forecast features)")
                    print("   May have been trained with buggy caching (20-step cache)")
                    print("   [NOT RECOMMENDED] Use earlier checkpoint (state_dim=900) or start fresh")
                else:
                    print("[ERROR] RISKY: This checkpoint is very recent with forecast features")
                    print("   Likely trained with buggy forecast caching")
                    print("   [NOT RECOMMENDED] Use checkpoint with state_dim=900 or start fresh")
            else:
                print(f"[WARN] Unknown state dimension: {state_dim}")
                print("   Cannot determine if safe to use")
        
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Check latest checkpoint
        models_dir = Path("models")
        checkpoints = sorted(
            [f for f in models_dir.glob("checkpoint_*.pt")],
            key=lambda x: int(x.stem.split('_')[1])
        )
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])
            print(f"Checking latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found")
            sys.exit(1)
    
    check_checkpoint(checkpoint_path)

