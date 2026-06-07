"""OpenRouter-провайдер. OpenAI-совместимый, нужен только API-ключ.

OpenRouter — агрегатор моделей (Anthropic, Google, Meta, …) с единым
ключом. По умолчанию используем дешёвую и быструю модель ``openai/gpt-4o-mini``,
но через ``OPENROUTER_MODEL`` можно переключить.
"""

from __future__ import annotations

import os
from typing import Iterable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import ChatMessage, env_int

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM:
    provider_id = "openrouter"

    # OpenRouter агрегирует разные модели; берём безопасный потолок 4000
    # (gpt-4o-mini держит 16k, claude-3.5-sonnet — 8k, llama-3.1-70b — 4k).
    # Чтобы крутнуть выше — OPENROUTER_MAX_TOKENS=8000.
    DEFAULT_MAX_OUTPUT = 4000
    # Окно gpt-4o-mini = 128k. Для остальных моделей обычно в районе
    # 8k–200k. OPENROUTER_CONTEXT_WINDOW подстроит под конкретную модель.
    DEFAULT_CONTEXT_WINDOW = 128_000

    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY не задан. Получите ключ на "
                "https://openrouter.ai/keys и пропишите в .env."
            )
        self._api_key = api_key.strip()
        self.model_label = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        # 60 сек не хватает бесплатным бэкендам (Llama 405B, Nemotron 120B):
        # очередь + длинная генерация легко дают 90+ сек.
        self._timeout = float(os.getenv("OPENROUTER_TIMEOUT", "180"))
        self.max_output_tokens = env_int(
            "OPENROUTER_MAX_TOKENS", self.DEFAULT_MAX_OUTPUT
        )
        self.context_window_tokens = env_int(
            "OPENROUTER_CONTEXT_WINDOW",
            self.DEFAULT_CONTEXT_WINDOW,
            hi=2_000_000,
        )

    # См. подробный комментарий о retry-баге в mistral.py: tenacity ловит
    # *исходные* httpx-исключения, поэтому внутренняя функция _chat_inner
    # их НЕ перехватывает и НЕ перепаковывает. Только финальная (после 3
    # неудачных попыток) ошибка попадает в chat() и оборачивается в
    # понятный RuntimeError.
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(
            (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError)
        ),
        reraise=True,
    )
    def _chat_inner(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        capped = min(max_tokens, self.max_output_tokens)
        body = {
            "model": self.model_label,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": capped,
        }
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    # OpenRouter рекомендует указывать X-Title — для красивой
                    # статистики в их кабинете.
                    "X-Title": "methodist_chat",
                },
                json=body,
            )
        if r.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter HTTP {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish = choice.get("finish_reason") or ""
        if finish == "length":
            text += (
                "\n\n[ОБРЕЗАНО: выбран весь лимит токенов max_tokens="
                f"{capped}, будет докомплектовано continuation-вызовом]"
            )
        return text

    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> str:
        msg_list = list(messages)
        try:
            return self._chat_inner(
                msg_list, temperature=temperature, max_tokens=max_tokens
            )
        except httpx.RemoteProtocolError as exc:
            raise RuntimeError(
                "OpenRouter оборвал соединение (RemoteProtocolError) после 3 "
                "попыток. Частые причины: бесплатный бэкенд перегружен, "
                "очередь, лимит реквестов в минуту. Попробуйте другую модель "
                f"в OPENROUTER_MODEL (сейчас {self.model_label!r}) или "
                f"поднимите OPENROUTER_TIMEOUT (сейчас {self._timeout:.0f}s)."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                f"OpenRouter не ответил за {self._timeout:.0f}s после 3 "
                "попыток. Поднимите OPENROUTER_TIMEOUT в .env или выберите "
                "быструю модель (openai/gpt-4o-mini, "
                "mistralai/mistral-small-3.1-24b-instruct:free)."
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Не удалось подключиться к openrouter.ai после 3 попыток. "
                "Проверьте интернет-соединение и доступность сервиса."
            ) from exc

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
