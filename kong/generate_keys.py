#!/usr/bin/env python3
"""
Generate secure API keys for Kong consumers
"""

import secrets
import sys

def generate_key(name: str) -> str:
    """Generate a secure random key"""
    return secrets.token_urlsafe(32)

if __name__ == "__main__":
    keys = {
        "REASONING_ENGINE_KEY": generate_key("reasoning-engine"),
        "SWARM_AGENT_KEY": generate_key("swarm-agent"),
        "QUERY_DEEPSEEK_KEY": generate_key("query-deepseek"),
        "ADMIN_KEY": generate_key("admin")
    }
    
    print("# Generated API Keys for Kong Consumers")
    print("# Save these keys securely and update kong.yml and .env file")
    print()
    for name, key in keys.items():
        print(f"{name}={key}")
    
    print("\n# Add these to your .env file:")
    print("# KONG_REASONING_ENGINE_KEY=...")
    print("# KONG_SWARM_AGENT_KEY=...")
    print("# KONG_QUERY_DEEPSEEK_KEY=...")
    print("# KONG_ADMIN_KEY=...")

