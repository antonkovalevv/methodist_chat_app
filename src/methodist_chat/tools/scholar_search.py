"""scholar_search — научные публикации (CrossRef / OpenAlex / arXiv / cyberleninka)."""

from __future__ import annotations

from typing import Any

from methodist_chat.infra.tools.scholar_search import (
    ScholarRequest,
    ScholarSearchTool,
)

from .base import Source, ToolOutput


class ScholarSearch:
    name = "scholar_search"
    description = (
        "Поиск реальных научных публикаций в CrossRef, OpenAlex, arXiv и "
        "Cyberleninka. Зови ПОСЛЕ rag_search — чтобы дополнить или "
        "обновить список литературы методиста современными публикациями "
        "(с DOI, авторами, годом, журналом). Поддерживает фильтр по году."
    )
    args_schema = {
        "query": "тема поиска (по-русски или по-английски)",
        "year_from": "минимальный год публикации (например, 2020). Опционально.",
        "year_to": "максимальный год публикации. Опционально.",
        "max_results": "сколько результатов вернуть (по умолчанию 8)",
    }

    def __init__(self) -> None:
        self._impl = ScholarSearchTool()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        query = (args.get("query") or "").strip()
        if not query:
            return ToolOutput(text="", error="Параметр 'query' не задан")
        max_results = min(int(args.get("max_results", 8)), 15)
        year_from = args.get("year_from")
        year_to = args.get("year_to")
        try:
            req = ScholarRequest(
                query=query,
                year_from=int(year_from) if year_from else None,
                year_to=int(year_to) if year_to else None,
                max_results=max_results,
            )
            res = self._impl.run(req)
        except Exception as e:
            return ToolOutput(text="", error=f"scholar_search: {e}")
        pubs = res.data.publications
        if not pubs:
            return ToolOutput(
                text=f"По запросу '{query}' научных публикаций не найдено."
            ).truncate()

        lines = [f"Найдено {len(pubs)} публикаций по запросу '{query}':\n"]
        sources: list[Source] = []
        for i, p in enumerate(pubs, 1):
            # Защита от None во ВСЕХ полях: scholar-движки иногда возвращают
            # частичные данные, и слайс по None даёт TypeError.
            authors_list = p.authors or []
            title = p.title or "(без названия)"
            container = p.container or ""
            abstract = p.abstract or ""
            doi = p.doi or ""
            url = p.url or ""

            authors = ", ".join(authors_list[:3]) + (
                "…" if len(authors_list) > 3 else ""
            )
            line = f"[{i}] {authors or 'без автора'} ({p.year or '?'}). «{title}»"
            if container:
                line += f" — {container}"
            if doi:
                line += f"\n    DOI: {doi}"
            elif url:
                line += f"\n    URL: {url}"
            line += f"\n    источник: {p.source_engine or '—'}"
            if abstract:
                line += f"\n    {abstract[:300]}"
            lines.append(line)
            lines.append("")
            sources.append(
                Source(
                    title=title,
                    url=url or (f"https://doi.org/{doi}" if doi else ""),
                    snippet=(abstract or container)[:300],
                    extra={
                        "authors": authors_list,
                        "year": p.year,
                        "doi": doi,
                        "engine": p.source_engine or "",
                    },
                )
            )
        return ToolOutput(text="\n".join(lines), sources=sources).truncate()
