"""
Kong Gateway Client Wrapper

Provides a unified interface for routing LLM requests through Kong Gateway.
Handles authentication, error handling, and request routing.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class KongProvider(Enum):
    """Kong route providers"""
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROK = "grok"
    OLLAMA = "ollama"


@dataclass
class KongConfig:
    """Kong Gateway configuration"""
    base_url: str = "http://localhost:8300"
    api_key: Optional[str] = None
    provider: KongProvider = KongProvider.OLLAMA
    timeout: int = 600


class KongClient:
    """
    Client for routing requests through Kong Gateway.
    
    Handles:
    - API key authentication
    - Request routing to Kong services
    - Error handling and retries
    - Response caching (via Kong proxy-cache plugin)
    """
    
    # Kong route mappings
    ROUTE_MAP = {
        KongProvider.ANTHROPIC: "/llm/anthropic/v1",
        KongProvider.DEEPSEEK: "/llm/deepseek/v1",
        KongProvider.GROK: "/llm/grok/v1",
        KongProvider.OLLAMA: "/llm/ollama/api"
    }
    
    def __init__(
        self,
        kong_base_url: str = "http://localhost:8300",
        api_key: str = None,
        provider: KongProvider = KongProvider.OLLAMA
    ):
        """
        Initialize Kong client.
        
        Args:
            kong_base_url: Kong Gateway proxy URL (default: http://localhost:8300)
            api_key: Kong consumer API key (required for authentication)
            provider: LLM provider to route to
        """
        self.base_url = kong_base_url.rstrip('/')
        self.provider = provider
        self.api_key = api_key or os.getenv("KONG_API_KEY")
        
        if not self.api_key:
            # Try to get provider-specific key from environment
            provider_keys = {
                KongProvider.ANTHROPIC: "KONG_ANTHROPIC_KEY",
                KongProvider.DEEPSEEK: "KONG_DEEPSEEK_KEY",
                KongProvider.GROK: "KONG_GROK_KEY",
                KongProvider.OLLAMA: "KONG_OLLAMA_KEY"
            }
            env_key = provider_keys.get(provider)
            if env_key:
                self.api_key = os.getenv(env_key)
        
        if not self.api_key:
            raise ValueError(
                f"Kong API key required. Set KONG_API_KEY environment variable "
                f"or pass api_key parameter. Provider: {provider.value}"
            )
    
    def _get_route(self) -> str:
        """Get Kong route for the provider"""
        route = self.ROUTE_MAP.get(self.provider)
        if not route:
            raise ValueError(f"Unknown provider: {self.provider}")
        return f"{self.base_url}{route}"
    
    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get request headers with Kong authentication"""
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        if additional_headers:
            headers.update(additional_headers)
        return headers
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[int] = None,
        keep_alive: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Send chat request through Kong Gateway.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Temperature parameter
            top_p: Top-p parameter
            stream: Whether to stream response
            timeout: Request timeout in seconds
            keep_alive: Keep model loaded (Ollama only)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Response text from LLM
        """
        route = self._get_route()
        
        # Build request URL based on provider
        if self.provider == KongProvider.OLLAMA:
            url = f"{route}/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            }
            if keep_alive:
                payload["keep_alive"] = keep_alive
        else:
            # Anthropic, DeepSeek, Grok use OpenAI-compatible format
            url = f"{route}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            payload.update(kwargs)
        
        headers = self._get_headers()
        
        try:
            if stream:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=None,
                    stream=True
                )
                response.raise_for_status()
                
                # Handle streaming response
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            if self.provider == KongProvider.OLLAMA:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    full_content += chunk["message"]["content"]
                                if chunk.get("done", False):
                                    break
                            else:
                                # OpenAI-compatible SSE format
                                if line.startswith(b"data: "):
                                    data_str = line[6:].decode('utf-8')
                                    if data_str.strip() == "[DONE]":
                                        break
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            full_content += delta["content"]
                                    if data.get("choices", [{}])[0].get("finish_reason") == "stop":
                                        break
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            continue
                
                return full_content
            else:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout or 600
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Parse response based on provider
                if self.provider == KongProvider.OLLAMA:
                    return result.get("message", {}).get("content", "")
                elif self.provider == KongProvider.ANTHROPIC:
                    # Anthropic uses different response format
                    if "content" in result:
                        # Anthropic direct format
                        content_blocks = result.get("content", [])
                        if content_blocks and len(content_blocks) > 0:
                            return content_blocks[0].get("text", "")
                    # Fallback to OpenAI-compatible
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    # DeepSeek, Grok use OpenAI-compatible format
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Kong authentication failed. Check API key.")
            elif e.response.status_code == 403:
                raise ValueError("Kong access denied. Check API key permissions.")
            elif e.response.status_code == 429:
                raise ValueError("Kong rate limit exceeded. Try again later.")
            else:
                error_msg = f"Kong Gateway error: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data.get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {e.response.text}"
                raise RuntimeError(error_msg)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to Kong Gateway timed out after {timeout or 600}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Kong Gateway: {e}")
    
    def get_cache_status(self, response: requests.Response) -> Optional[str]:
        """
        Get cache status from Kong response headers.
        
        Args:
            response: HTTP response object
        
        Returns:
            Cache status: "HIT", "MISS", or None
        """
        return response.headers.get("X-Cache-Status")

