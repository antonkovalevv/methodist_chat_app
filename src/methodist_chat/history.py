"""Оптимизация истории чата.

Модель работы:

* В UI ``session_state.history`` хранит список ``ChatTurn`` —
  пары (user_text, assistant_text). Промежуточные tool-вызовы НЕ
  попадают в history (они «съедаются» внутри одного хода).
* Перед каждым новым запросом к LLM мы вызываем ``compress_history``,
  который:
    1. Оставляет N последних пар дословно (по умолчанию 4).
    2. Если старых пар > порога — просит LLM сжать их в одну строку
       (``rolling_summary``).
    3. Если общий объём всё ещё превышает токен-бюджет — обрезает
       старейшие сжатые куски.

Возвращается «компактная» цепочка ChatMessage, которую агент подмешает
к системному промпту.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from .llm import ChatMessage, LLMProvider
from .prompts import SUMMARY_PROMPT


_log = logging.getLogger(__name__)


# Эвристика: 1 токен ≈ 3 символа русского текста (на латинице — 4). Берём
# консервативно 3, чтобы не переоценить бюджет.
_CHARS_PER_TOKEN = 3


@dataclass
class ChatTurn:
    user: str
    assistant: str


@dataclass
class CompressedHistory:
    """Результат сжатия для подмешивания в промпт LLM."""

    summary: str = ""
    recent_turns: list[ChatTurn] = field(default_factory=list)

    def to_messages(self) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        if self.summary:
            msgs.append(
                ChatMessage(
                    role="system",
                    content=f"[Контекст ранее в диалоге]\n{self.summary}",
                )
            )
        for t in self.recent_turns:
            msgs.append(ChatMessage(role="user", content=t.user))
            msgs.append(ChatMessage(role="assistant", content=t.assistant))
        return msgs


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compress_history(
    turns: Iterable[ChatTurn],
    *,
    llm: LLMProvider | None = None,
    keep_recent: int = 4,
    token_budget: int = 2000,
    summarize_threshold: int = 6,
    previous_summary: str = "",
) -> CompressedHistory:
    """Главная функция оптимизации истории.

    Параметры:
        turns: полная история сессии (включая текущий, ещё не отвеченный
            ход — НЕ передавай его сюда; передавай только завершённые).
        llm: используется для сжатия старых ходов. Если ``None`` —
            старые ходы просто отбрасываются (на тестах удобно).
        keep_recent: сколько последних ходов передавать дословно.
        token_budget: жёсткий потолок суммарных токенов всей цепочки
            (summary + recent_turns).
        summarize_threshold: если в истории больше этого числа ходов,
            обновляем rolling summary через LLM.
        previous_summary: ранее посчитанный summary (если есть) — он
            попадает в промпт сжатия как «уже было сжато ранее…».
    """
    turns_list = list(turns)
    if not turns_list:
        return CompressedHistory()

    if len(turns_list) <= keep_recent:
        return _enforce_budget(
            CompressedHistory(summary=previous_summary, recent_turns=turns_list),
            token_budget=token_budget,
        )

    recent = turns_list[-keep_recent:]
    old = turns_list[:-keep_recent]

    if not old:
        return _enforce_budget(
            CompressedHistory(summary=previous_summary, recent_turns=recent),
            token_budget=token_budget,
        )

    # Решение: пересжимать ли summary этим ходом.
    should_summarize = (
        llm is not None
        and len(turns_list) >= summarize_threshold
    )
    summary = previous_summary
    if should_summarize:
        summary = _summarize(llm, old, previous_summary)

    return _enforce_budget(
        CompressedHistory(summary=summary, recent_turns=recent),
        token_budget=token_budget,
    )


def _summarize(
    llm: LLMProvider,
    old_turns: list[ChatTurn],
    previous_summary: str,
) -> str:
    """Вызов LLM для свёртки старых ходов в одну строку."""
    history_block: list[str] = []
    if previous_summary:
        history_block.append(f"[Ранее уже сжато] {previous_summary}\n")
    for i, t in enumerate(old_turns, 1):
        history_block.append(f"[ход {i}] методист: {t.user}")
        history_block.append(f"          ассистент: {t.assistant}")
    payload = "\n".join(history_block)
    t0 = time.monotonic()
    _log.info(
        "  summarise history: → LLM call (%d old turns, %d chars payload)",
        len(old_turns),
        len(payload),
    )
    try:
        out = llm.quick(
            system=SUMMARY_PROMPT,
            user=payload,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception as e:
        # Если свёртка падает — не блокируем диалог: просто оставим как было.
        _log.warning(
            "  summarise history: ✕ failed after %.1fs: %s",
            time.monotonic() - t0,
            e,
        )
        return previous_summary
    dt = time.monotonic() - t0
    out = (out or "").strip()[:1500]  # потолок длины самого summary
    _log.info(
        "  summarise history: ← LLM (%d chars in %.1fs)", len(out), dt
    )
    return out


def _enforce_budget(
    h: CompressedHistory, *, token_budget: int
) -> CompressedHistory:
    """Жёсткий потолок: режем самое старое (recent_turns), потом сам summary."""
    while _tokens_of(h) > token_budget and len(h.recent_turns) > 1:
        h.recent_turns.pop(0)
    if _tokens_of(h) > token_budget and h.summary:
        # обрезаем summary до того, что влезает
        allowed_chars = max(200, token_budget * _CHARS_PER_TOKEN - _tokens_text_chars(h))
        h.summary = h.summary[:allowed_chars]
    return h


def _tokens_text_chars(h: CompressedHistory) -> int:
    return sum(len(t.user) + len(t.assistant) for t in h.recent_turns)


def _tokens_of(h: CompressedHistory) -> int:
    """Покомпонентная оценка токенов: summary + каждая ChatTurn отдельно.

    Применять estimate_tokens к каждому куску по-отдельности корректнее, чем
    к их суммарной длине: иначе мы недооцениваем округление вверх (max(1, ...))
    на коротких ходах.
    """
    return estimate_tokens(h.summary) + sum(
        estimate_tokens(t.user) + estimate_tokens(t.assistant) for t in h.recent_turns
    )
