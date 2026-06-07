"""Общий веб-поиск. Дефолт — DuckDuckGo (бесплатно, без ключа).

Tavily опционально подключается, если в .env задан TAVILY_API_KEY.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from methodist_chat.infra.config import get_settings
from methodist_chat.infra.logging import get_logger
from .base import Source, Tool, ToolResult
from .registry import register

logger = get_logger(__name__)


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    region: str = "ru-ru"


class WebHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = "duckduckgo"


class WebSearchResponse(BaseModel):
    hits: list[WebHit] = Field(default_factory=list)


def _search_ddg(query: str, max_results: int, region: str) -> list[WebHit]:
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

    out: list[WebHit] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, region=region):
                out.append(
                    WebHit(
                        title=r.get("title", ""),
                        url=r.get("href", "") or r.get("url", ""),
                        snippet=r.get("body", "") or r.get("description", ""),
                        source="duckduckgo",
                    )
                )
    except Exception as e:
        logger.warning(f"DDG: {e}")
    return out


def _search_tavily(query: str, max_results: int, api_key: str) -> list[WebHit]:
    try:
        from tavily import TavilyClient
    except ImportError:  # pragma: no cover
        return []

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        out: list[WebHit] = []
        for r in response.get("results", []):
            out.append(
                WebHit(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source="tavily",
                )
            )
        return out
    except Exception as e:  # pragma: no cover
        logger.warning(f"Tavily: {e}")
        return []


@register("web_search")
class WebSearchTool(Tool[WebSearchRequest, WebSearchResponse]):
    description = "Веб-поиск через DuckDuckGo (по умолчанию) или Tavily, если доступен ключ."

    def run(self, params: WebSearchRequest) -> ToolResult[WebSearchResponse]:
        settings = get_settings()
        hits: list[WebHit] = []
        if settings.tavily_api_key:
            hits = _search_tavily(params.query, params.max_results, settings.tavily_api_key)
        if not hits:
            hits = _search_ddg(params.query, params.max_results, params.region)

        sources = [
            Source(
                kind="web",
                title=h.title,
                url=h.url,
                snippet=h.snippet[:300],
                fetched_at=datetime.utcnow(),
                extra={"engine": h.source},
            )
            for h in hits
        ]
        return ToolResult(
            data=WebSearchResponse(hits=hits),
            sources=sources,
            log=[f"q='{params.query[:60]}' hits={len(hits)}"],
        )
