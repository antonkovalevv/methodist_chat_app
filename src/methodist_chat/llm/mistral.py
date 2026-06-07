"""Mistral-провайдер. OpenAI-совместимый ``/v1/chat/completions``."""

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

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralLLM:
    provider_id = "mistral"

    # На практике Mistral Small отвечает ~30 сек на 4000 токенов и ~80 сек на
    # 8000 — у их шлюза часто рвётся соединение раньше, поэтому дефолт здесь
    # консервативный (4000). Длинные ответы дочитывает continuation-loop в
    # agent.py. Перебивается MISTRAL_MAX_TOKENS.
    DEFAULT_MAX_OUTPUT = 4000
    # Mistral Small держит 32k контекст; Large — 128k. Под другую модель
    # выставьте MISTRAL_CONTEXT_WINDOW.
    DEFAULT_CONTEXT_WINDOW = 32_000

    def __init__(self) -> None:
        api_key = os.getenv("MISTRAL_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY не задан. Получите ключ на "
                "https://console.mistral.ai/api-keys/ и пропишите в .env."
            )
        self._api_key = api_key.strip()
        self.model_label = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        # 60 сек не хватает для длинных ответов — шлюз Mistral закрывает
        # соединение, httpx бросает RemoteProtocolError. 180 сек — комфортно.
        self._timeout = float(os.getenv("MISTRAL_TIMEOUT", "180"))
        self.max_output_tokens = env_int(
            "MISTRAL_MAX_TOKENS", self.DEFAULT_MAX_OUTPUT
        )
        self.context_window_tokens = env_int(
            "MISTRAL_CONTEXT_WINDOW",
            self.DEFAULT_CONTEXT_WINDOW,
            hi=500_000,
        )

    # Внутренняя функция с tenacity-retry. Принципиально важно: НЕ ловим
    # здесь httpx.RemoteProtocolError / ReadTimeout / ConnectError — иначе
    # tenacity их не увидит (он ловит ровно эти классы, а перепакованный
    # RuntimeError для него «уже не наша ошибка» и retry не делается).
    # Был баг: чат падал с первой попытки без retry. После исправления
    # tenacity делает 3 попытки с экспоненциальным backoff, и только если
    # все три неудачны — внешний chat() перепаковывает в RuntimeError с
    # понятным сообщением.
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
        # Аппаратный потолок провайдера — нельзя просить больше, чем держит
        # модель/тариф; иначе шлюз молча обрывает соединение.
        capped = min(max_tokens, self.max_output_tokens)
        body = {
            "model": self.model_label,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": capped,
        }
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(
                MISTRAL_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Mistral HTTP {r.status_code}: {r.text[:500]}")
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
                "Mistral оборвал соединение (RemoteProtocolError) после 3 "
                "попыток. Это не лимит токенов, а таймаут шлюза провайдера. "
                f"Попробуйте уменьшить MISTRAL_MAX_TOKENS (сейчас "
                f"{self.max_output_tokens}) или поднять MISTRAL_TIMEOUT "
                f"(сейчас {self._timeout:.0f}s). Также часто помогает "
                "переключиться на mistral-small-latest в .env."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                f"Mistral не ответил за {self._timeout:.0f}s после 3 попыток. "
                "Поднимите MISTRAL_TIMEOUT в .env или уменьшите MISTRAL_MAX_TOKENS."
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Не удалось подключиться к api.mistral.ai после 3 попыток. "
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
