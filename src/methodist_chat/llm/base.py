"""Единый интерфейс LLM-провайдера для methodist_chat.

Все три провайдера (GigaChat, OpenRouter, Mistral) поддерживают
OpenAI-совместимый протокол ``chat.completions``. Поэтому один общий
интерфейс из 4 методов покрывает их всех:

* ``chat(messages, **kw) -> str`` — основной метод.
* ``quick(system, user, **kw) -> str`` — сахар для одноразового вызова.
* ``model_label`` — что показывать в UI.
* ``provider_id`` — ``"gigachat" | "openrouter" | "mistral"``.
* ``max_output_tokens`` — реальный потолок ответа конкретной модели.
  Агент берёт лимит отсюда, чтобы не подбирать его «вслепую» — у каждого
  провайдера он свой (GigaChat-Lite — 2000, GPT-4o-mini — 4000+,
  Mistral — 8000+). Перекрывается env-переменными
  ``GIGACHAT_MAX_TOKENS`` / ``OPENROUTER_MAX_TOKENS`` / ``MISTRAL_MAX_TOKENS``.
* ``context_window_tokens`` — суммарный размер окна модели (input+output).
  Используется агентом, чтобы автоматически подобрать **размер чанков
  документа** под конкретную модель: для модели с 32k окна нельзя кидать
  весь PDF на 200 КБ — придётся резать. Перекрывается env-переменными
  ``*_CONTEXT_WINDOW``.

Если модель упёрлась в ``max_output_tokens`` посреди финального ответа,
сервер дописывает в текст маркер ``[ОБРЕЗАНО: ...]``. Агент видит маркер
и сам делает continuation-вызов: «продолжи финальный ответ с того
места, где оборвался». Так пользователь получает ответ целиком, даже
если он не помещается в одно обращение к модели.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Protocol


def env_int(name: str, default: int, *, lo: int = 256, hi: int = 32000) -> int:
    """Парсит env-переменную как int с защитой от мусора и диапазоном."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    if lo <= v <= hi:
        return v
    return default


_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f", ""}


def env_bool(name: str, default: bool = False) -> bool:
    """Парсит env-переменную как bool. Терпим к 1/0, true/false, yes/no, on/off."""
    raw = (os.getenv(name) or "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return default


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class LLMProvider(Protocol):
    """Минимальный интерфейс провайдера, на который опирается агент."""

    provider_id: str
    model_label: str
    max_output_tokens: int
    context_window_tokens: int

    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str: ...

    def quick(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str: ...


# Эвристика «токены → символы» для русского текста (подходит и для GigaChat,
# и для OpenAI/Mistral tokenizer-ов: средний рос. токен — 2–4 символа).
TOKENS_TO_CHARS_RU = 3


# Маркер обрезки. Серверный wrapper дописывает его, если провайдер вернул
# finish_reason=length. Агент по нему делает continuation-вызов.
TRUNCATION_MARKER = "[ОБРЕЗАНО:"


def context_window_chars(provider: "LLMProvider") -> int:
    return provider.context_window_tokens * TOKENS_TO_CHARS_RU


def attachment_budget(provider: "LLMProvider") -> int:
    """Сколько символов прикреплённого документа можно безопасно класть в
    контекст (head_preview + read_attachment).

    Из общего окна провайдера вычитаем «накладные»:
      • системный промпт + сжатая история диалога (~20 000 симв.),
      • буфер ReAct-цикла (Thought/Action/Observation × 4) (~20 000 симв.),
      • запас под результаты инструментов и оверхед (~10 000 симв.),
      • выходной бюджет (max_output_tokens × 3 ~= 12 000 симв. для 4k токенов).

    Если посчитать всё это, у Mistral Small / GigaChat-Lite (32k окно,
    ≈96k символов) на документ остаётся ~30–40k симв., у gpt-4o-mini
    (128k окно) — много больше, но обрезаем сверху, чтобы модель не
    «терялась» в 100k+ контексте. Границы: [6 000, 50 000].
    """
    total = context_window_chars(provider)
    reserved = 50_000 + provider.max_output_tokens * TOKENS_TO_CHARS_RU
    free = total - reserved
    return max(6_000, min(50_000, free))
