"""web_search — DuckDuckGo (по умолчанию) или Tavily."""

from __future__ import annotations

from typing import Any

from methodist_chat.infra.tools.web_search import WebSearchTool, WebSearchRequest

from .base import Source, ToolOutput


class WebSearch:
    name = "web_search"
    description = (
        "Поиск в интернете (DuckDuckGo / Tavily). Зови, когда нужны "
        "СВЕЖИЕ факты, которых не может быть в локальной RAG-базе: "
        "релиз-ноты, EOL-даты, новости, текущие версии ПО, недавние "
        "статьи и документы, обновления нормативной базы, проверка "
        "ссылок. Также подходит, когда rag_search вернул мало или "
        "слабо релевантные хиты. Возвращает заголовки, URL и сниппеты."
    )
    args_schema = {
        "query": "строка-запрос (по-русски или по-английски)",
        "max_results": "сколько результатов вернуть (по умолчанию 5, максимум 10)",
    }

    def __init__(self) -> None:
        self._impl = WebSearchTool()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        query = (args.get("query") or "").strip()
        if not query:
            return ToolOutput(text="", error="Параметр 'query' не задан")
        max_results = min(int(args.get("max_results", 5)), 10)
        try:
            res = self._impl.run(WebSearchRequest(query=query, max_results=max_results))
        except Exception as e:
            return ToolOutput(text="", error=f"web_search: {e}")
        hits = res.data.hits
        if not hits:
            return ToolOutput(text=f"По запросу '{query}' ничего не найдено.").truncate()
        lines = [f"Найдено {len(hits)} результатов по запросу '{query}':\n"]
        sources: list[Source] = []
        for i, h in enumerate(hits, 1):
            # Защита от None: некоторые движки могут вернуть None в snippet.
            snippet = (h.snippet or "")[:300]
            lines.append(f"[{i}] {h.title or '(без заголовка)'}")
            lines.append(f"    URL: {h.url or '—'}")
            if snippet:
                lines.append(f"    {snippet}")
            lines.append("")
            sources.append(
                Source(title=h.title or "", url=h.url or "", snippet=snippet)
            )
        return ToolOutput(text="\n".join(lines), sources=sources).truncate()
