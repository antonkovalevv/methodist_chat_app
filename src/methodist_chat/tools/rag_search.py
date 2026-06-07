"""rag_search — поиск по локальной БД (нормативка, ГОСТ, ФГОС).

Использует существующий ChromaDB-индекс ``constants_search``. Если БД пуста
или не инициализирована — возвращает пустой результат, без ошибки. Это
важно: на чистой машине без --build индекса агент должен работать,
просто без RAG.

Фильтрация по релевантности: фрагменты с релевантностью ниже
``RAG_TOOL_MIN_RELEVANCE`` (env, дефолт 0.20) отбрасываются ДО возврата
модели. Это режет очевидный шум, не давая ему отъесть context window.
Порог здесь мягче, чем у preflight (`RAG_PREFLIGHT_MIN_RELEVANCE`,
дефолт 0.45): тут модель уже явно вызвала инструмент и ожидает увидеть
хоть что-то — лучше отдать слабые хиты с пометкой, чем «нет совпадений».
"""

from __future__ import annotations

import logging
import os
from typing import Any

from methodist_chat.infra.tools.vector_db import ConstantsSearchTool, SearchRequest

from .base import Source, ToolOutput

_log = logging.getLogger("methodist_chat.rag_search")


# Порог релевантности по умолчанию. Хиты ниже него полностью отбрасываются.
# Идея: 0.0 — это «вообще не связано», 0.30 — «слабое совпадение, может
# пригодиться как контекст», 0.45+ — «уверенное совпадение». Режем только
# совсем мусор (< 0.20), всё остальное отдаём модели с пометкой о силе.
DEFAULT_TOOL_MIN_RELEVANCE = 0.20


def _tool_min_relevance() -> float:
    """Читает RAG_TOOL_MIN_RELEVANCE из env с защитой от мусора."""
    raw = os.getenv("RAG_TOOL_MIN_RELEVANCE", "")
    if not raw:
        return DEFAULT_TOOL_MIN_RELEVANCE
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_TOOL_MIN_RELEVANCE
    if 0.0 <= v <= 1.0:
        return v
    return DEFAULT_TOOL_MIN_RELEVANCE


class RagSearch:
    name = "rag_search"
    description = (
        "ПРИОРИТЕТ #1 — локальная база знаний методиста. Здесь лежат "
        "нормативные документы (ФГОС, ГОСТ, шаблоны оформления, выдержки "
        "приказов) И ВСЕ файлы, которые методист сам загрузил в раздел "
        "«База знаний (RAG)»: РПД, ФОС, лекции, методички, программы "
        "курсов. Это источник истины первого порядка — ВСЕГДА вызывай "
        "rag_search в самом начале ответа на содержательный запрос "
        "методиста, до scholar_search/web_search и до собственных знаний. "
        "Делай 2-3 разных запроса по теме (термины, синонимы, код "
        "направления), а не один — это резко повышает покрытие."
    )
    args_schema = {
        "query": "запрос (по-русски). Делай содержательный — фразу или 3-5 ключевых терминов",
        "n_results": "сколько фрагментов вернуть (по умолчанию 6, максимум 10)",
    }

    def __init__(self) -> None:
        self._impl = ConstantsSearchTool()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        query = (args.get("query") or "").strip()
        if not query:
            return ToolOutput(text="", error="Параметр 'query' не задан")
        # Дефолт согласован с описанием в args_schema: 6.
        n = min(int(args.get("n_results", 6)), 10)
        try:
            res = self._impl.run(SearchRequest(query=query, n_results=n))
        except Exception as e:
            return ToolOutput(text="", error=f"rag_search: {e}")
        raw_hits = res.data.hits

        # Фильтрация: отбрасываем заведомый шум. Логируем, сколько срезали,
        # чтобы было видно в трейсе и при отладке.
        threshold = _tool_min_relevance()
        hits = [h for h in raw_hits if h.relevance >= threshold]
        dropped = len(raw_hits) - len(hits)
        if dropped:
            _log.info(
                "rag_search: фильтрация по relevance>=%.2f отбросила %d/%d "
                "фрагментов (query=%r)",
                threshold,
                dropped,
                len(raw_hits),
                query[:80],
            )

        if not hits:
            # БД может быть пуста (raw_hits=0) ИЛИ всё что нашлось — мусор
            # (dropped>0 и hits=0). Сообщение одинаковое — для модели это
            # один и тот же сигнал «иди в другие источники».
            extra = ""
            if dropped:
                max_raw = max((h.relevance for h in raw_hits), default=0.0)
                extra = (
                    f" Лучшее совпадение в БД — {max_raw:.2f}, что ниже "
                    f"порога релевантности {threshold:.2f}."
                )
            return ToolOutput(
                text=(
                    f"В локальной БД нормативки нет совпадений по запросу '{query}'.{extra} "
                    "Попробуй web_search или scholar_search."
                )
            ).truncate()

        max_relevance = max((h.relevance for h in hits), default=0.0)
        header = f"Найдено {len(hits)} фрагментов в локальной БД (макс. релевантность: {max_relevance:.2f})"
        if dropped:
            header += f" [отброшено {dropped} слабых по порогу {threshold:.2f}]"
        if max_relevance < 0.30:
            header += (
                ".\n[слабое совпадение] Релевантность пограничная — фрагменты "
                "могут быть полезны как контекст, но не как единственный "
                "источник нормативного факта. Если данных мало — допиши "
                "через scholar_search/web_search."
            )
        lines = [header + "\n"]
        sources: list[Source] = []
        for i, h in enumerate(hits, 1):
            title = h.metadata.get("description") or h.metadata.get("type") or "константа"
            origin = h.metadata.get("source") or ""
            origin_tag = " [пользовательский файл]" if origin == "user_upload" else ""
            lines.append(f"[{i}] {title}{origin_tag} (релевантность: {h.relevance:.2f})")
            lines.append(f"    {h.text[:400]}")
            lines.append("")
            sources.append(
                Source(title=str(title), snippet=h.text[:300], extra=h.metadata)
            )
        return ToolOutput(text="\n".join(lines), sources=sources).truncate()
