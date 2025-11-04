"""
LLM Provider Abstraction Layer
Supports multiple providers: Ollama, DeepSeek Cloud API, Grok (xAI)
"""

import json
import requests
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    DEEPSEEK_CLOUD = "deepseek_cloud"
    GROK = "grok"


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> str:
        """Send chat messages and get response"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> str:
        """Call Ollama API"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        
        try:
            if stream:
                response = requests.post(url, json=payload, timeout=None, stream=True)
                response.raise_for_status()
                
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                full_content += chunk["message"]["content"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                return full_content
            else:
                response = requests.post(url, json=payload, timeout=timeout or 600)
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
                
        except requests.exceptions.Timeout:
            print(f"Warning: Ollama request timed out after {timeout or 600}s")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return ""


class DeepSeekCloudProvider(BaseLLMProvider):
    """DeepSeek Cloud API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> str:
        """Call DeepSeek Cloud API"""
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        try:
            if stream:
                response = requests.post(url, json=payload, headers=headers, timeout=None, stream=True)
                response.raise_for_status()
                
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            # DeepSeek uses SSE format: "data: {...}"
                            if line.startswith(b"data: "):
                                data = json.loads(line[6:])  # Skip "data: " prefix
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                                if data.get("choices", [{}])[0].get("finish_reason") == "stop":
                                    break
                        except json.JSONDecodeError:
                            continue
                
                return full_content
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout or 120)
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except requests.exceptions.Timeout:
            print(f"Warning: DeepSeek Cloud request timed out after {timeout or 120}s")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error calling DeepSeek Cloud: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return ""


class GrokProvider(BaseLLMProvider):
    """Grok (xAI) API provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.x.ai"):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-beta",
        temperature: float = 0.3,
        top_p: float = 0.9,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> str:
        """Call Grok API"""
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        try:
            if stream:
                response = requests.post(url, json=payload, headers=headers, timeout=None, stream=True)
                response.raise_for_status()
                
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            # Grok uses SSE format: "data: {...}"
                            if line.startswith(b"data: "):
                                line_data = line[6:].decode('utf-8')
                                if line_data == "[DONE]":
                                    break
                                data = json.loads(line_data)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                                if data.get("choices", [{}])[0].get("finish_reason") == "stop":
                                    break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                
                return full_content
            else:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout or 120)
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except requests.exceptions.Timeout:
            print(f"Warning: Grok request timed out after {timeout or 120}s")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error calling Grok: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return ""


def get_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
    """
    Factory function to get the appropriate provider
    
    Args:
        provider_type: "ollama", "deepseek_cloud", or "grok"
        **kwargs: Provider-specific configuration
            - For Ollama: base_url (optional, default: http://localhost:11434)
            - For DeepSeek Cloud: api_key (required), base_url (optional)
            - For Grok: api_key (required), base_url (optional)
    
    Returns:
        BaseLLMProvider instance
    """
    provider_type = provider_type.lower()
    
    if provider_type == LLMProvider.OLLAMA.value:
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaProvider(base_url=base_url)
    
    elif provider_type == LLMProvider.DEEPSEEK_CLOUD.value:
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for DeepSeek Cloud provider")
        base_url = kwargs.get("base_url", "https://api.deepseek.com")
        return DeepSeekCloudProvider(api_key=api_key, base_url=base_url)
    
    elif provider_type == LLMProvider.GROK.value:
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("api_key is required for Grok provider")
        base_url = kwargs.get("base_url", "https://api.x.ai")
        return GrokProvider(api_key=api_key, base_url=base_url)
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: ollama, deepseek_cloud, grok")

