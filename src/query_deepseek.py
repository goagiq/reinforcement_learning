"""
Query Ollama DeepSeek model for RL trading strategy recommendations.
This script will be used to get AI-assisted recommendations for the NT8 RL project.
"""

import json
import requests
from typing import Dict, List, Optional


class OllamaClient:
    """Client for interacting with Ollama API (directly or through Kong)."""
    
    def __init__(self, base_url: str = "http://localhost:11434", use_kong: bool = False, kong_api_key: str = None):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama base URL (ignored if use_kong=True)
            use_kong: Route requests through Kong Gateway
            kong_api_key: Kong consumer API key (required if use_kong=True)
        """
        import os
        self.use_kong = use_kong
        if use_kong:
            self.base_url = os.getenv("KONG_BASE_URL", "http://localhost:8300")
            self.kong_api_key = kong_api_key or os.getenv("KONG_OLLAMA_KEY") or os.getenv("KONG_API_KEY")
            if not self.kong_api_key:
                raise ValueError("Kong API key required when use_kong=True. Set KONG_OLLAMA_KEY or KONG_API_KEY")
        else:
            self.base_url = base_url
            self.kong_api_key = None
        self.model = "deepseek-r1:8b"
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = True, timeout: int = 600, keep_alive: str = "10m") -> str:
        """
        Send chat messages to Ollama model with streaming support.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response (default: True for better UX)
            timeout: Timeout in seconds (default: 600 for DeepSeek-R1 reasoning)
            keep_alive: Keep model loaded in memory (default: "10m" for 10 minutes)
        
        Returns:
            Complete response text
        """
        if self.use_kong:
            url = f"{self.base_url}/llm/ollama/api/chat"
        else:
            url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,  # Keep model pre-loaded for faster responses
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 4000,  # Limit response length for faster responses
            }
        }
        
        try:
            headers = {}
            if self.use_kong and self.kong_api_key:
                headers["apikey"] = self.kong_api_key
            
            if stream:
                # Stream response for better UX with longer timeout
                # DeepSeek-R1 needs time for reasoning, so we disable timeout for stream
                print("⏳ Waiting for DeepSeek-R1 reasoning (this may take 2-5 minutes)...")
                print("   Streaming response as it's generated:\n")
                
                response = requests.post(
                    url, 
                    json=payload,
                    headers=headers,
                    timeout=None,  # No timeout - let it stream
                    stream=True
                )
                response.raise_for_status()
                
                full_content = ""
                last_print_time = 0
                chunk_count = 0
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            
                            # Handle content chunks
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                print(content, end="", flush=True)
                                full_content += content
                                chunk_count += 1
                            
                            # Check if done
                            if chunk.get("done", False):
                                break
                            
                            # Progress indicator for very long waits
                            if chunk_count == 0 and "message" not in chunk:
                                # Model is thinking/reasoning
                                pass
                                
                        except json.JSONDecodeError:
                            continue
                
                print()  # New line after streaming
                
                if not full_content:
                    return "Warning: Received empty response. Model may still be processing."
                
                return full_content
            else:
                # Non-streaming response with longer timeout
                print("⏳ Generating response (this may take 2-5 minutes for DeepSeek-R1)...")
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except requests.exceptions.Timeout:
            print(f"\n⚠️  Request timed out after {timeout} seconds.")
            print("   DeepSeek-R1 models take longer due to reasoning. Try:")
            print("   1. Wait longer (reasoning models can take 5+ minutes)")
            print("   2. Use streaming mode (default) for progress")
            print("   3. Check if Ollama is processing (check Ollama logs)")
            return "Error: Request timed out. DeepSeek-R1 reasoning takes time."
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Error querying Ollama: {e}")
            print("   Make sure Ollama is running: ollama serve")
            return f"Error: {str(e)}"
    


def get_rl_strategy_recommendations() -> str:
    """
    Query DeepSeek for RL trading strategy recommendations using chat() API.
    
    Returns:
        AI-generated recommendations
    """
    client = OllamaClient()
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in reinforcement learning and algorithmic trading. Provide clear, actionable recommendations."
        },
        {
            "role": "user",
            "content": """I'm developing a NinjaTrader 8 trading strategy using PyTorch reinforcement learning, focusing on price action and volume analysis.

Please provide detailed recommendations on:

1. **State Space Design**: What features from price action and volume should be included? How should they be normalized?
2. **Action Space**: Should I use discrete actions (buy/sell/hold) or continuous position sizing? Why?
3. **Reward Function**: How should I design the reward function to balance profit, risk, and transaction costs?
4. **RL Algorithm Selection**: Which algorithm (PPO, DQN, SAC, A3C) is best for this use case? Why?
5. **Hyperparameter Tuning**: What key hyperparameters should I focus on?
6. **Continuous Learning**: How should I structure the fine-tuning pipeline for my DeepSeek model to continuously learn from trading outcomes?

Provide specific, actionable recommendations with technical details."""
        }
    ]
    
    print(f"Querying {client.model} for RL strategy recommendations...\n")
    response = client.chat(messages, stream=True)
    
    if response.startswith("Error:"):
        return response
    
    return response


def get_reasoning_recommendations() -> str:
    """
    Query DeepSeek for reasoning and reflection architecture recommendations.
    
    Returns:
        AI-generated reasoning recommendations
    """
    client = OllamaClient()
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in AI reasoning systems, particularly DeepSeek-R1 reasoning models for decision-making applications."
        },
        {
            "role": "user",
            "content": """I'm building a reinforcement learning trading strategy that integrates DeepSeek-R1:8b for deep reasoning and reflection.

The system needs to:
1. Analyze RL model recommendations before executing trades (pre-trade reasoning)
2. Reflect on completed trades to extract lessons (post-trade reflection)
3. Continuously monitor market regime changes
4. Provide explainable decisions with reasoning chains

Please recommend:
1. Optimal reasoning prompt structure for DeepSeek-R1 in trading context
2. How to structure chain-of-thought reasoning for market analysis
3. Integration patterns between RL model and reasoning layer
4. Confidence scoring when combining RL output with reasoning
5. Methods to prevent reasoning overhead from delaying trade execution
6. How to use reasoning insights to improve RL model learning
7. Best practices for prompt engineering with DeepSeek-R1:8b
8. Strategies for handling conflicts between RL and reasoning recommendations

Provide detailed, technical recommendations with examples."""
        }
    ]
    
    print(f"Querying {client.model} for reasoning architecture recommendations...\n")
    response = client.chat(messages, stream=True)
    
    if response.startswith("Error:"):
        return response
    
    return response


def get_finetuning_recommendations() -> str:
    """
    Query DeepSeek for fine-tuning recommendations for continuous learning.
    
    Returns:
        AI-generated fine-tuning recommendations
    """
    client = OllamaClient()
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in LLM fine-tuning, particularly for financial/trading applications."
        },
        {
            "role": "user",
            "content": """I want to fine-tune DeepSeek-R1:8b for continuous learning in a trading strategy context.

The model will receive:
- Market state features (price action, volume, indicators)
- Trading outcomes (profit/loss, win rate, drawdown)
- Market regime information

I need the model to:
1. Provide trading recommendations based on market state
2. Learn from successful vs failed trades
3. Adapt to changing market conditions
4. Run efficiently (8B model on local hardware)

Please recommend:
1. Fine-tuning approach (LoRA, QLoRA, or full fine-tuning)
2. Training data format and structure
3. Training schedule (how often to fine-tune)
4. Techniques to prevent catastrophic forgetting
5. How to balance model updates vs stability
6. Specific hyperparameters for DeepSeek-R1:8b fine-tuning

Provide detailed, technical recommendations."""
        }
    ]
    
    print(f"Querying {client.model} for fine-tuning recommendations...\n")
    response = client.chat(messages, stream=True)
    
    if response.startswith("Error:"):
        return response
    
    return response


if __name__ == "__main__":
    print("=" * 80)
    print("NT8 RL Strategy - AI Recommendations from DeepSeek-R1:8b")
    print("=" * 80)
    print()
    
    # Check if model is available
    try:
        check_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = check_response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if "deepseek-r1:8b" not in model_names:
            print("⚠️  Warning: deepseek-r1:8b not found in Ollama models.")
            print(f"Available models: {', '.join(model_names)}")
            print("Please wait for the model to finish downloading.")
            print()
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama: {e}")
        print("Make sure Ollama is running on http://localhost:11434")
        print()
    
    print("\n" + "="*80)
    print("NOTE: DeepSeek-R1:8b uses deep reasoning which can take 2-5 minutes per query.")
    print("Responses will stream in real-time. Please be patient.")
    print("="*80 + "\n")
    
    print("1. Getting RL Strategy Recommendations...")
    print("-" * 80)
    rl_recommendations = get_rl_strategy_recommendations()
    if not rl_recommendations.startswith("Error:"):
        print("\n✅ RL Strategy Recommendations received!")
    print()
    
    print("\n2. Getting Reasoning Architecture Recommendations...")
    print("-" * 80)
    reasoning_recommendations = get_reasoning_recommendations()
    if not reasoning_recommendations.startswith("Error:"):
        print("\n✅ Reasoning Recommendations received!")
    print()
    
    print("\n3. Getting Fine-Tuning Recommendations...")
    print("-" * 80)
    finetuning_recommendations = get_finetuning_recommendations()
    if not finetuning_recommendations.startswith("Error:"):
        print("\n✅ Fine-Tuning Recommendations received!")
    print()
    
    # Save to file
    output = {
        "rl_recommendations": rl_recommendations,
        "reasoning_recommendations": reasoning_recommendations,
        "finetuning_recommendations": finetuning_recommendations
    }
    
    with open("docs/recommendations_deepseek.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("=" * 80)
    print("Recommendations saved to: docs/recommendations_deepseek.json")
    print("=" * 80)

