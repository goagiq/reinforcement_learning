"""
Service helpers for AI-generated capability analyses and tooltips.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.capability_registry import CapabilityMetadata, get_capability
from src.llm_providers import get_provider, LLMProvider


DEFAULT_LOCALE = "en-US"
STORE_ROOT = Path("data/ai_analysis")


class AIAnalysisStore:
    """Filesystem-backed store for AI capability analyses."""

    def __init__(self, root: Path = STORE_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, capability_id: str, locale: str, user_id: Optional[str]) -> Path:
        sanitized_locale = locale.replace("/", "_")
        user_segment = user_id or "global"
        return self.root / capability_id.replace(".", "/") / sanitized_locale / f"{user_segment}.json"

    def load(
        self,
        capability_id: str,
        locale: str,
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        path = self._path(capability_id, locale, user_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save(
        self,
        capability_id: str,
        locale: str,
        user_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        path = self._path(capability_id, locale, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


store = AIAnalysisStore()


def ensure_capability(capability_id: str) -> CapabilityMetadata:
    capability = get_capability(capability_id)
    if not capability:
        raise HTTPException(status_code=404, detail=f"Capability '{capability_id}' not found")
    return capability


def build_prompt(
    capability: CapabilityMetadata,
    context: Dict[str, Any],
    locale: str,
) -> Dict[str, str]:
    """Create prompts for analysis and tooltip generation."""
    base_context = {
        "locale": locale,
        "timestamp": datetime.utcnow().isoformat(),
    }
    full_context = {**base_context, **context}

    def safe_format(template: str) -> str:
        try:
            return template.format(**full_context)
        except KeyError:
            # Fall back to template if placeholders missing
            return template

    return {
        "analysis": safe_format(capability.prompt_template),
        "tooltip": safe_format(capability.tooltip_template),
    }


def call_llm(prompt: str, provider_hint: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Execute LLM call with graceful degradation if provider is unavailable.

    provider_hint: optional provider name to bias selection (ollama, deepseek_cloud, etc.).
    """
    provider_type = provider_hint or os.getenv("AI_CAPABILITY_PROVIDER", "ollama")
    model_name = model or os.getenv("AI_CAPABILITY_MODEL", "deepseek-r1:8b")

    use_kong = os.getenv("USE_KONG_LLM", "false").lower() == "true"
    provider_kwargs: Dict[str, Any] = {"use_kong": use_kong}

    if provider_type == LLMProvider.DEEPSEEK_CLOUD.value:
        provider_kwargs["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    elif provider_type == LLMProvider.GROK.value:
        provider_kwargs["api_key"] = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    elif provider_type == LLMProvider.ANTHROPIC.value:
        provider_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        provider_kwargs["max_output_tokens"] = 2048
    elif provider_type == LLMProvider.OLLAMA.value:
        provider_kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        provider = get_provider(provider_type, **provider_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM provider setup failed: {exc}") from exc

    messages = [
        {
            "role": "system",
            "content": "You are an expert trading assistant who writes concise, actionable guidance.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = provider.chat(messages=messages, model=model_name, temperature=0.4, top_p=0.9)
        if not response:
            raise RuntimeError("Empty response from language model")
        return response.strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc


def moderate_output(text: str) -> bool:
    """
    Simple heuristic moderation. Returns True if content passes.
    """
    banned_keywords = ["<script", "DROP TABLE", "DELETE FROM", "http://", "https://"]
    lower = text.lower()
    for keyword in banned_keywords:
        if keyword.lower() in lower:
            return False
    return True


def generate_analysis(
    capability_id: str,
    locale: str,
    user_id: Optional[str],
    context: Dict[str, Any],
    force_refresh: bool = False,
    provider_hint: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    capability = ensure_capability(capability_id)

    locale = locale or DEFAULT_LOCALE
    cache_hit = None if force_refresh else store.load(capability_id, locale, user_id)
    if cache_hit:
        return {"cached": True, "data": cache_hit}

    prompts = build_prompt(capability, context, locale)

    analysis_text = call_llm(prompts["analysis"], provider_hint=provider_hint, model=model)
    tooltip_text = call_llm(prompts["tooltip"], provider_hint=provider_hint, model=model)

    if not moderate_output(analysis_text) or not moderate_output(tooltip_text):
        raise HTTPException(status_code=400, detail="Generated content failed moderation")

    payload = {
        "capability_id": capability_id,
        "locale": locale,
        "analysis": analysis_text,
        "tooltip": tooltip_text,
        "generated_at": datetime.utcnow().isoformat(),
        "context_snapshot": context,
    }
    store.save(capability_id, locale, user_id, payload)
    return {"cached": False, "data": payload}


def record_feedback(
    capability_id: str,
    locale: str,
    user_id: Optional[str],
    feedback: Dict[str, Any],
) -> None:
    """
    Append structured feedback next to the cached analysis.
    """
    capability = ensure_capability(capability_id)
    locale = locale or DEFAULT_LOCALE

    existing = store.load(capability_id, locale, user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="No generated analysis found to attach feedback")

    feedback_entry = {
        "submitted_at": datetime.utcnow().isoformat(),
        "rating": feedback.get("rating"),
        "comment": feedback.get("comment"),
        "source": feedback.get("source", "frontend"),
    }

    feedback_list = existing.setdefault("feedback", [])
    feedback_list.append(feedback_entry)
    store.save(capability_id, locale, user_id, existing)


