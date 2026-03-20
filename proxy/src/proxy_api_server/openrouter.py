"""OpenRouter client utilities used by the proxy API server."""

from __future__ import annotations

from devs_utilities.openrouter import (
    ChatCompletionResult,
    OpenRouterClient as SharedOpenRouterClient,
    OpenRouterConfig,
    OpenRouterError,
    ToolCall,
)

from proxy_api_server.config import Settings


class OpenRouterClient(SharedOpenRouterClient):
    """Settings-aware wrapper around the shared OpenRouter client."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            OpenRouterConfig(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                model=settings.openrouter_model,
                timeout_seconds=settings.openrouter_timeout_seconds,
                site_url=settings.openrouter_app_url,
                site_name=settings.openrouter_app_title,
            )
        )
