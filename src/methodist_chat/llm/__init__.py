"""LLM-провайдеры: GigaChat, OpenRouter, Mistral.

Загружаются лениво — `make_provider("openrouter")` импортирует только
нужный модуль, чтобы отсутствие ключей других провайдеров не ломало
загрузку.
"""

from __future__ import annotations

from .base import ChatMessage, LLMProvider

__all__ = ["ChatMessage", "LLMProvider", "make_provider", "AVAILABLE_PROVIDERS"]


AVAILABLE_PROVIDERS = ("gigachat", "openrouter", "mistral")


def make_provider(provider_id: str) -> LLMProvider:
    """Фабрика провайдера по строковому id.

    Поднимает ``ValueError`` если id неизвестен. Может поднять
    ``ValueError``/``RuntimeError`` если у выбранного провайдера нет ключа.
    """
    pid = (provider_id or "").lower().strip()
    if pid == "gigachat":
        from .gigachat import GigaChatLLM

        return GigaChatLLM()
    if pid == "openrouter":
        from .openrouter import OpenRouterLLM

        return OpenRouterLLM()
    if pid == "mistral":
        from .mistral import MistralLLM

        return MistralLLM()
    raise ValueError(
        f"Неизвестный провайдер '{provider_id}'. "
        f"Допустимые: {', '.join(AVAILABLE_PROVIDERS)}."
    )
