"""
Benchmark Swarm Performance

Runs performance tests and generates reports.
"""

import argparse
import yaml
import time
import asyncio
from datetime import datetime
from pathlib import Path
import json

from src.agentic_swarm import SwarmOrchestrator
from src.agentic_swarm.cost_tracker import CostTracker


def benchmark_swarm(config_path: str, iterations: int = 10):
    """
    Benchmark swarm performance.
    
    Args:
        config_path: Path to config file
        iterations: Number of iterations to run
    """
    print("=" * 60)
    print("Swarm Performance Benchmark")
    print("=" * 60)
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize orchestrator
    print("Initializing swarm orchestrator...")
    orchestrator = SwarmOrchestrator(config)
    
    # Initialize cost tracker
    cost_tracker = CostTracker()
    
    # Mock market data
    market_data = {
        "price_data": {
            "open": 5000.0,
            "high": 5010.0,
            "low": 4990.0,
            "close": 5005.0
        },
        "volume_data": {"volume": 1000000},
        "indicators": {},
        "market_regime": "trending",
        "timestamp": datetime.now().isoformat()
    }
    
    rl_recommendation = {
        "action": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong bullish signal"
    }
    
    # Run benchmarks
    execution_times = []
    costs = []
    success_count = 0
    
    print(f"\nRunning {iterations} iterations...")
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)
        
        start_time = time.time()
        result = orchestrator.analyze_sync(
            market_data=market_data,
            rl_recommendation=rl_recommendation,
            current_position=0.0
        )
        execution_time = time.time() - start_time
        
        execution_times.append(execution_time)
        
        if result.get("status") == "success":
            success_count += 1
            print(f"✓ ({execution_time:.2f}s)")
        else:
            print(f"✗ ({execution_time:.2f}s) - {result.get('status', 'unknown')}")
        
        # Simulate cost tracking
        cost_tracker.log_llm_call(
            agent_name="analyst",
            provider="ollama",
            model="deepseek-r1:8b",
            tokens_input=500,
            tokens_output=200,
            duration_seconds=2.0
        )
    
    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)
    
    success_rate = (success_count / iterations) * 100
    
    # Get cost statistics
    cost_stats = cost_tracker.get_statistics()
    
    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Iterations: {iterations}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"\nExecution Time:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Min: {min_time:.2f}s")
    print(f"  Max: {max_time:.2f}s")
    print(f"\nCost Statistics:")
    print(f"  Total Cost: ${cost_stats['total_cost']:.4f}")
    print(f"  Avg Cost per Call: ${cost_stats['avg_cost_per_call']:.4f}")
    print(f"  Total API Calls: {cost_stats['total_calls']}")
    
    # Status check
    print(f"\nSwarm Status:")
    status = orchestrator.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "success_rate": success_rate,
        "execution_times": {
            "average": avg_time,
            "min": min_time,
            "max": max_time,
            "all": execution_times
        },
        "cost_statistics": cost_stats,
        "swarm_status": status
    }
    
    output_path = Path("test_output") / f"swarm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Performance assessment
    print("\n" + "=" * 60)
    print("Performance Assessment")
    print("=" * 60)
    
    if avg_time < 5.0:
        print("✅ Excellent: Average execution time < 5s")
    elif avg_time < 10.0:
        print("✅ Good: Average execution time < 10s")
    elif avg_time < 20.0:
        print("⚠️  Acceptable: Average execution time < 20s")
    else:
        print("❌ Slow: Average execution time >= 20s")
    
    if success_rate >= 95.0:
        print("✅ Excellent: Success rate >= 95%")
    elif success_rate >= 90.0:
        print("✅ Good: Success rate >= 90%")
    elif success_rate >= 80.0:
        print("⚠️  Acceptable: Success rate >= 80%")
    else:
        print("❌ Poor: Success rate < 80%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark swarm performance")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run"
    )
    
    args = parser.parse_args()
    benchmark_swarm(args.config, args.iterations)

