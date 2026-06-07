"""read_attachment — чтение фрагмента загруженного методистом файла.

UI кладёт **полный** распарсенный текст вложения в ``session_state``.
В системный промпт передаётся карточка документа + первые ~10 000 символов
(``head``). Если методист спрашивает что-то конкретное (раздел 5.2,
литература, термины), агент зовёт этот инструмент с режимом ``query`` и
получает релевантный фрагмент через простой keyword-поиск (BM25 не
поднимаем — для прототипа на 200k символов достаточно скользящего окна
с подсчётом совпадений).

Зачем простой keyword вместо embeddings: один прогон embeddings на 50k
символов — это ~2-5 секунд + загрузка модели в память. Для прототипа на
защите это лишняя сложность. Keyword-режим даёт за 100 мс «ткнуть в
нужное место документа», и это видно в трейсе.
"""

from __future__ import annotations

import re
from typing import Any

from .base import ToolOutput


_DEFAULT_WINDOW_SIZE = 1500   # символов в одном окне (база)
_DEFAULT_WINDOW_STRIDE = 1000  # шаг сдвига (с перекрытием)
_DEFAULT_MAX_WINDOWS = 4       # вернуть не больше N лучших окон
_DEFAULT_MAX_CHARS = 6000      # суммарный потолок ответа инструмента


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)


def _score_window(window_tokens: list[str], query_tokens: list[str]) -> int:
    if not query_tokens:
        return 0
    s = 0
    qset = set(query_tokens)
    for t in window_tokens:
        if t in qset:
            s += 1
    return s


def keyword_search(
    text: str,
    query: str,
    max_windows: int = _DEFAULT_MAX_WINDOWS,
    *,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    window_stride: int = _DEFAULT_WINDOW_STRIDE,
) -> list[str]:
    """Возвращает топ-N окон документа, наиболее релевантных запросу."""
    if not text or not query:
        return []
    qtok = _tokenize(query)
    if not qtok:
        return []

    windows: list[tuple[int, int, str]] = []  # (score, start, snippet)
    pos = 0
    n = len(text)
    while pos < n:
        chunk = text[pos : pos + window_size]
        ctok = _tokenize(chunk)
        score = _score_window(ctok, qtok)
        if score > 0:
            windows.append((score, pos, chunk))
        pos += window_stride
        if pos >= n:
            break
    windows.sort(key=lambda w: (-w[0], w[1]))
    chosen = windows[:max_windows]
    chosen.sort(key=lambda w: w[1])  # вернуть в порядке появления в документе
    return [w[2] for w in chosen]


class ReadAttachment:
    """Доступ к загруженному документу.

    Создаётся per-session с привязкой к тексту документа в session_state.
    """

    name = "read_attachment"
    description = (
        "Читает фрагменты загруженного методистом файла. Режимы: "
        "'head' — начало документа (первые ~10 000 символов), "
        "'tail' — конец, 'query' — наиболее релевантные запросу окна "
        "(простой keyword-поиск). Используй, когда нужно увидеть, что "
        "конкретно сказано в документе, прежде чем отвечать или искать в "
        "интернете."
    )
    args_schema = {
        "mode": "'head' | 'tail' | 'query' (по умолчанию 'query')",
        "query": "запрос — обязательно при mode='query'",
        "max_chars": (
            "максимум символов в ответе (по умолчанию подбирается под "
            "контекстное окно модели)"
        ),
    }

    def __init__(
        self,
        attachment_text: str = "",
        filename: str = "",
        *,
        chars_budget: int = _DEFAULT_MAX_CHARS,
    ) -> None:
        self._text = attachment_text or ""
        self._filename = filename or ""
        # Per-provider бюджет: на маленькой модели (GigaChat-Lite, 32k окно)
        # отдадим ~6000 символов, на gpt-4o-mini (128k) — до ~30000.
        self.chars_budget = max(1500, int(chars_budget))

    def update(self, attachment_text: str, filename: str = "") -> None:
        self._text = attachment_text or ""
        self._filename = filename or self._filename

    def _window_size(self) -> int:
        # Размер одного «чанка» в keyword-search масштабируем под бюджет.
        # На малом окне — 1500 (как было), на большом — до 4000.
        return min(4000, max(1500, self.chars_budget // 4))

    def _window_stride(self) -> int:
        return int(self._window_size() * 0.66)

    def run(self, args: dict[str, Any]) -> ToolOutput:
        if not self._text:
            return ToolOutput(
                text="",
                error="Документ не загружен. Попроси методиста прикрепить файл.",
            )
        mode = (args.get("mode") or "query").strip().lower()
        max_chars = int(args.get("max_chars", self.chars_budget))
        # Жёсткий потолок — бюджет провайдера.
        max_chars = min(max_chars, self.chars_budget)
        if mode == "head":
            chunk = self._text[:max_chars]
            return ToolOutput(
                text=f"Файл: {self._filename}\nНачало документа:\n\n{chunk}"
            ).truncate(max_chars + 300)
        if mode == "tail":
            chunk = self._text[-max_chars:]
            return ToolOutput(
                text=f"Файл: {self._filename}\nКонец документа:\n\n{chunk}"
            ).truncate(max_chars + 300)
        # mode == "query"
        query = (args.get("query") or "").strip()
        if not query:
            return ToolOutput(text="", error="При mode='query' нужен 'query'")
        windows = keyword_search(
            self._text,
            query,
            window_size=self._window_size(),
            window_stride=self._window_stride(),
        )
        if not windows:
            return ToolOutput(
                text=(
                    f"Файл: {self._filename}\n"
                    f"По запросу '{query}' релевантных фрагментов не найдено."
                )
            )
        joined = "\n\n---\n\n".join(windows)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "\n…[truncated]"
        return ToolOutput(
            text=(
                f"Файл: {self._filename}\nЗапрос: {query}\n"
                f"Найдено {len(windows)} релевантных фрагментов:\n\n{joined}"
            )
        )
