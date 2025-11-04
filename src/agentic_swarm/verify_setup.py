"""
Verification script for Phase 1 setup.

Run this to verify that all dependencies and infrastructure are set up correctly.
"""

import sys
from pathlib import Path

def verify_setup():
    """Verify Phase 1 setup."""
    print("=" * 60)
    print("Phase 1 Setup Verification")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check Strands Agents
    print("\n1. Checking Strands Agents SDK...")
    try:
        import strands
        from strands import Agent
        from strands.multiagent import Swarm
        print("   ✅ Strands Agents SDK installed")
    except ImportError as e:
        errors.append(f"Strands Agents SDK not installed: {e}")
        print(f"   ❌ Strands Agents SDK not installed")
        print(f"      Install with: pip install strands")
    
    # Check directories
    print("\n2. Checking directory structure...")
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir,
        base_dir / "agents",
        base_dir / "tools",
        Path(__file__).parent.parent / "data_sources"
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"   ✅ {dir_path.name}/")
        else:
            errors.append(f"Directory missing: {dir_path}")
            print(f"   ❌ {dir_path.name}/ missing")
    
    # Check modules
    print("\n3. Checking modules...")
    modules = [
        "src.agentic_swarm.shared_context",
        "src.agentic_swarm.base_agent",
        "src.agentic_swarm.config_loader",
        "src.agentic_swarm.swarm_orchestrator",
        "src.data_sources.market_data",
        "src.data_sources.sentiment_sources",
        "src.data_sources.cache"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            errors.append(f"Module import failed: {module} - {e}")
            print(f"   ❌ {module}")
    
    # Check config
    print("\n4. Checking configuration...")
    try:
        import yaml
        config_path = Path("configs/train_config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "agentic_swarm" in config:
                    print("   ✅ agentic_swarm config found")
                else:
                    warnings.append("agentic_swarm config not found in train_config.yaml")
                    print("   ⚠️  agentic_swarm config not found")
        else:
            errors.append("configs/train_config.yaml not found")
            print("   ❌ configs/train_config.yaml not found")
    except Exception as e:
        errors.append(f"Config check failed: {e}")
        print(f"   ❌ Config check failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ {len(errors)} error(s) found:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("\n✅ Phase 1 setup verification complete!")
    print("   Ready to proceed to Phase 2 (Individual Agents)")
    return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)

