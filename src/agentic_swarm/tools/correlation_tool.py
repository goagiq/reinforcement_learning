"""
Correlation Tool

Tool for calculating correlations between instruments.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def calculate_pairwise_correlation(
    prices1: List[float],
    prices2: List[float],
    window: int = 20
) -> float:
    """
    Calculate rolling correlation between two price series.
    
    Args:
        prices1: First price series
        prices2: Second price series
        window: Rolling window size
    
    Returns:
        Correlation coefficient (-1.0 to 1.0)
    """
    if len(prices1) < window or len(prices2) < window:
        return 0.0
    
    # Use last N values
    p1 = np.array(prices1[-window:])
    p2 = np.array(prices2[-window:])
    
    # Calculate correlation
    corr = np.corrcoef(p1, p2)[0, 1]
    
    return float(corr) if not np.isnan(corr) else 0.0


def calculate_correlation_matrix(
    price_data: Dict[str, List[float]],
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple instruments.
    
    Args:
        price_data: Dict of instrument -> price series
        window: Rolling window size
    
    Returns:
        Correlation matrix DataFrame
    """
    instruments = list(price_data.keys())
    n = len(instruments)
    
    # Create correlation matrix
    corr_matrix = np.eye(n)
    
    for i, inst1 in enumerate(instruments):
        for j, inst2 in enumerate(instruments):
            if i != j:
                corr = calculate_pairwise_correlation(
                    price_data[inst1],
                    price_data[inst2],
                    window
                )
                corr_matrix[i, j] = corr
    
    return pd.DataFrame(corr_matrix, index=instruments, columns=instruments)


def detect_divergence(
    base_prices: List[float],
    comparison_prices: Dict[str, List[float]],
    threshold: float = 0.1,
    window: int = 20
) -> Dict[str, Any]:
    """
    Detect divergence between base instrument and comparison instruments.
    
    Args:
        base_prices: Base instrument price series
        comparison_prices: Dict of instrument -> price series
        threshold: Divergence threshold
        window: Rolling window size
    
    Returns:
        Dict with divergence signals
    """
    signals = {}
    
    for instrument, prices in comparison_prices.items():
        corr = calculate_pairwise_correlation(base_prices, prices, window)
        
        if abs(corr) < (1.0 - threshold):
            signals[instrument] = "divergence"
        elif abs(corr) > (1.0 - threshold * 0.5):
            signals[instrument] = "convergence"
        else:
            signals[instrument] = "normal"
    
    return {
        "signals": signals,
        "threshold": threshold,
        "window": window
    }

