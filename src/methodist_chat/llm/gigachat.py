"""GigaChat-провайдер для methodist_chat.

Тонкий wrapper поверх ``gigachat_provider.GigaChatProvider`` —
OAuth/SSL/ретраи. Адаптирует к нашему интерфейсу ``LLMProvider``.
"""

from __future__ import annotations

import os
from typing import Iterable

from methodist_chat.llm.gigachat_provider import GigaChatProvider as _GC

from .base import ChatMessage, env_bool, env_int


class GigaChatLLM:
    provider_id = "gigachat"

    # GigaChat-Lite жёстко режет ~2048 токенов на ответ; Pro до ~4000.
    # Дефолт 2000 — безопасный для Lite. Перебивается GIGACHAT_MAX_TOKENS.
    # Дефолтная модель — GigaChat (Lite). Чтобы поднять потолок до Pro,
    # пропишите GIGACHAT_MODEL=GigaChat-Pro и GIGACHAT_MAX_TOKENS=4000.
    DEFAULT_MAX_OUTPUT = 2000
    # Окно у GigaChat-Lite — 32k токенов; Pro — 131k. Берём безопасный
    # минимум; пользователь может прописать GIGACHAT_CONTEXT_WINDOW=131072.
    DEFAULT_CONTEXT_WINDOW = 32_000

    def __init__(self) -> None:
        auth_key = os.getenv("GIGACHAT_AUTH_KEY") or ""
        scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        model = os.getenv("GIGACHAT_MODEL", "GigaChat")
        verify_ssl = env_bool("GIGACHAT_VERIFY_SSL", default=False)
        self._impl = _GC(
            auth_key=auth_key, scope=scope, model=model, verify_ssl=verify_ssl
        )
        self.model_label = self._impl.model
        self.max_output_tokens = env_int(
            "GIGACHAT_MAX_TOKENS", self.DEFAULT_MAX_OUTPUT
        )
        self.context_window_tokens = env_int(
            "GIGACHAT_CONTEXT_WINDOW",
            self.DEFAULT_CONTEXT_WINDOW,
            hi=200_000,
        )

    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> str:
        # Аппаратный потолок GigaChat-Lite — 2048; больше — провайдер вернёт
        # 400. Заодно защита от того, что агент случайно попросит больше.
        capped = min(max_tokens, self.max_output_tokens)
        text = self._impl.chat(messages, temperature=temperature, max_tokens=capped)
        # GigaChat-провайдер не возвращает finish_reason из ответа API.
        # Эвристика «упёрся в лимит» = «длинный ответ И не похоже, что
        # модель его сама закончила»:
        #   1. длина ответа в токенах >= 97% от max_tokens (раньше было 95%,
        #      что давало много false-positive на нормально завершившихся
        #      длинных ответах);
        #   2. последний непустой символ — НЕ терминальная пунктуация
        #      (.!?…»"`)).
        # Только когда оба условия выполнены, ставим маркер обрезки и
        # запускаем continuation-цикл.
        approx_tokens = max(1, len(text) // 3)  # 3 chars/token (RU)
        stripped = text.rstrip()
        last = stripped[-1] if stripped else ""
        looks_complete = last in '.!?…»"\'`)»]'
        if approx_tokens >= int(capped * 0.97) and not looks_complete:
            text += (
                f"\n\n[ОБРЕЗАНО: упёрся в лимит max_tokens={capped}, "
                "будет докомплектовано continuation-вызовом]"
            )
        return text

    def quick(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        return self.chat(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
