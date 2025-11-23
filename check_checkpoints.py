"""Quick script to list available checkpoints"""
import os
from pathlib import Path

models_dir = Path("models")
if not models_dir.exists():
    print("Models directory not found")
    exit(1)

files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
checkpoints = sorted(
    [f for f in files if f.startswith('checkpoint_')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

print("="*60)
print("AVAILABLE CHECKPOINTS")
print("="*60)
print(f"\nTotal checkpoints: {len(checkpoints)}")
print(f"\nFirst 10 checkpoints:")
for cp in checkpoints[:10]:
    print(f"  {cp}")

print(f"\nMilestone checkpoints:")
milestones = [100000, 500000, 1000000, 2000000, 3000000, 4000000]
for milestone in milestones:
    cp_name = f"checkpoint_{milestone}.pt"
    if cp_name in checkpoints:
        print(f"  [OK] {cp_name}")
    else:
        # Find closest
        closest = None
        min_diff = float('inf')
        for cp in checkpoints:
            steps = int(cp.split('_')[1].split('.')[0])
            diff = abs(steps - milestone)
            if diff < min_diff and steps <= milestone + 10000:
                min_diff = diff
                closest = (cp, steps)
        if closest:
            print(f"  [~] {milestone:,} -> {closest[0]} ({closest[1]:,} steps)")

print(f"\nLatest checkpoint: {checkpoints[-1] if checkpoints else 'None'}")

print(f"\nOther model files:")
other = [f for f in files if not f.startswith('checkpoint_')]
for f in other:
    print(f"  {f}")
