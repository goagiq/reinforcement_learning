"""
Generate Synthetic Data for Supervised Pre-training

This script generates synthetic market data with clear trading patterns
that will help the supervised pre-training learn useful signals.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.synthetic_data_generator import generate_synthetic_training_data
from src.utils.colors import success, info

if __name__ == "__main__":
    print(info("\nGenerating synthetic training data for supervised pre-training..."))
    
    generate_synthetic_training_data(
        output_dir="data/raw",
        instrument="ES",
        timeframes=[1, 5, 15],
        n_bars_per_tf={
            1: 20000,   # 20k bars of 1min (more data for better learning)
            5: 10000,   # 10k bars of 5min
            15: 5000    # 5k bars of 15min
        },
        seed=42  # Reproducible
    )
    
    print(success("\n[OK] Synthetic data generation complete!"))
    print(info("You can now run supervised pre-training with this data."))

