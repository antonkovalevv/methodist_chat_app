"""Поиск реальных научных публикаций.

Источники (все бесплатные, по убыванию приоритета):
  1. CrossRef REST API   — DOI + метаданные, без регистрации (mailto в User-Agent).
  2. OpenAlex            — крупнейший открытый граф работ; mailto-политика.
  3. arXiv API           — препринты, без регистрации.
  4. cyberleninka.ru     — русскоязычные статьи, парсинг по странице поиска.

Каждая запись возвращается в виде типизированного `Publication` с DOI/URL,
авторами, годом, журналом и краткой аннотацией. Это вход для `citation.py`,
который строит ссылку по ГОСТ Р 7.0.100-2018.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from methodist_chat.infra.config import get_settings
from methodist_chat.infra.logging import get_logger
from .base import Source, Tool, ToolResult
from .registry import register

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ScholarRequest(BaseModel):
    query: str
    year_from: int | None = None
    year_to: int | None = None
    max_results: int = 10
    sources: list[str] = Field(default_factory=lambda: ["crossref", "openalex", "cyberleninka"])
    lang: str = "ru"


class Publication(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    container: str = ""        # журнал/издательство/конференция
    publisher: str = ""
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str = ""
    type: str = "article"      # article | book | proceedings | preprint | other
    source_engine: str = ""    # crossref | openalex | arxiv | cyberleninka
    raw: dict[str, Any] = Field(default_factory=dict)

    def cite_short(self) -> str:
        bits = []
        if self.authors:
            bits.append(self.authors[0])
        if self.year:
            bits.append(str(self.year))
        if self.title:
            bits.append(f'«{self.title[:80]}»')
        return " ".join(bits)


class ScholarResponse(BaseModel):
    publications: list[Publication] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _crossref_query(query: str, max_results: int, mailto: str, year_from: int | None, year_to: int | None) -> list[Publication]:
    headers = {"User-Agent": f"methodist-assistant/0.1 (mailto:{mailto or 'noreply@example.com'})"}
    params: dict[str, Any] = {
        "query": query,
        "rows": max_results,
        "select": "DOI,title,author,container-title,issued,publisher,volume,issue,page,abstract,URL,type",
    }
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{year_from}")
    if year_to:
        filters.append(f"until-pub-date:{year_to}")
    if filters:
        params["filter"] = ",".join(filters)

    r = httpx.get("https://api.crossref.org/works", params=params, headers=headers, timeout=20.0)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    out: list[Publication] = []
    for it in items:
        title = (it.get("title") or [""])[0]
        authors = []
        for a in it.get("author", []) or []:
            name = " ".join(filter(None, [a.get("family"), a.get("given")]))
            if name.strip():
                authors.append(name.strip())
        issued = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        year = issued[0] if issued and isinstance(issued[0], int) else None
        out.append(
            Publication(
                title=title,
                authors=authors,
                year=year,
                container=(it.get("container-title") or [""])[0],
                publisher=it.get("publisher", "") or "",
                volume=it.get("volume"),
                issue=it.get("issue"),
                pages=it.get("page"),
                doi=it.get("DOI"),
                url=it.get("URL"),
                abstract=(it.get("abstract") or "")[:1000],
                type=it.get("type", "article"),
                source_engine="crossref",
                raw=it,
            )
        )
    return out


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _openalex_query(query: str, max_results: int, mailto: str, year_from: int | None, year_to: int | None) -> list[Publication]:
    params: dict[str, Any] = {
        "search": query,
        "per_page": max_results,
        "select": "id,doi,display_name,authorships,publication_year,primary_location,abstract_inverted_index,type,biblio",
    }
    if mailto:
        params["mailto"] = mailto
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    r = httpx.get("https://api.openalex.org/works", params=params, timeout=20.0)
    r.raise_for_status()
    items = r.json().get("results", [])
    out: list[Publication] = []
    for it in items:
        authors = []
        for a in it.get("authorships", []) or []:
            name = (a.get("author") or {}).get("display_name") or ""
            if name:
                authors.append(name)
        primary_location = it.get("primary_location") or {}
        source_block = primary_location.get("source") or {}
        venue = source_block.get("display_name", "")
        publisher = source_block.get("host_organization_name", "") or ""
        biblio = it.get("biblio") or {}
        # OpenAlex кодирует абстракт как inverted_index — раскрутим его
        abstract = _decode_inverted(it.get("abstract_inverted_index") or {})

        out.append(
            Publication(
                title=it.get("display_name", "") or "",
                authors=authors,
                year=it.get("publication_year"),
                container=venue,
                publisher=publisher,
                volume=biblio.get("volume"),
                issue=biblio.get("issue"),
                pages=(
                    f"{biblio.get('first_page')}–{biblio.get('last_page')}"
                    if biblio.get("first_page") else None
                ),
                doi=(it.get("doi") or "").replace("https://doi.org/", "") or None,
                url=it.get("id"),
                abstract=abstract[:1000],
                type=it.get("type", "article"),
                source_engine="openalex",
                raw=it,
            )
        )
    return out


def _decode_inverted(inv: dict[str, list[int]]) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def _arxiv_query(query: str, max_results: int) -> list[Publication]:
    try:
        import arxiv
    except ImportError:  # pragma: no cover
        return []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[Publication] = []
        for r in search.results():
            out.append(
                Publication(
                    title=r.title.strip(),
                    authors=[a.name for a in r.authors],
                    year=r.published.year if r.published else None,
                    container="arXiv preprint",
                    publisher="arXiv",
                    doi=r.doi,
                    url=r.entry_id,
                    abstract=(r.summary or "").strip()[:1000],
                    type="preprint",
                    source_engine="arxiv",
                )
            )
        return out
    except Exception as e:  # pragma: no cover
        logger.warning(f"arXiv: {e}")
        return []


# ---------------------------------------------------------------------------
# cyberleninka.ru — открытая русскоязычная база, есть страницы поиска
# ---------------------------------------------------------------------------


def _cyberleninka_query(query: str, max_results: int) -> list[Publication]:
    """Парсим страницу поиска Cyberleninka. У них нет официального API, но HTML стабильный."""
    try:
        url = f"https://cyberleninka.ru/search?q={quote_plus(query)}"
        r = httpx.get(url, headers={"User-Agent": "methodist-assistant/0.1"}, timeout=20.0)
        r.raise_for_status()
    except Exception as e:  # pragma: no cover
        logger.warning(f"cyberleninka GET: {e}")
        return []

    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    items = soup.select("ul.list li") or soup.select("article")
    out: list[Publication] = []
    for li in items[: max_results * 2]:  # с запасом, часть может отвалиться
        a = li.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a["href"]
        if href.startswith("/"):
            href = "https://cyberleninka.ru" + href
        # автор и год часто рядом, но без гарантий
        meta_text = li.get_text(" ", strip=True)
        year = None
        for token in meta_text.split():
            if token.isdigit() and 1990 <= int(token) <= datetime.utcnow().year + 1:
                year = int(token)
                break
        out.append(
            Publication(
                title=title,
                year=year,
                url=href,
                container="Cyberleninka",
                publisher="КиберЛенинка",
                source_engine="cyberleninka",
                type="article",
            )
        )
        if len(out) >= max_results:
            break
    return out


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register("scholar_search")
class ScholarSearchTool(Tool[ScholarRequest, ScholarResponse]):
    description = "Реальные публикации из CrossRef + OpenAlex + arXiv + cyberleninka."

    def run(self, params: ScholarRequest) -> ToolResult[ScholarResponse]:
        settings = get_settings()
        results: list[Publication] = []
        log: list[str] = []
        per_source = max(1, params.max_results)

        if "crossref" in params.sources:
            try:
                items = _crossref_query(
                    params.query, per_source, settings.crossref_mailto,
                    params.year_from, params.year_to,
                )
                results.extend(items)
                log.append(f"crossref={len(items)}")
            except Exception as e:
                log.append(f"crossref FAIL: {e}")

        if "openalex" in params.sources:
            try:
                items = _openalex_query(
                    params.query, per_source, settings.openalex_mailto,
                    params.year_from, params.year_to,
                )
                results.extend(items)
                log.append(f"openalex={len(items)}")
            except Exception as e:
                log.append(f"openalex FAIL: {e}")

        if "arxiv" in params.sources:
            items = _arxiv_query(params.query, per_source)
            results.extend(items)
            log.append(f"arxiv={len(items)}")

        if "cyberleninka" in params.sources:
            items = _cyberleninka_query(params.query, per_source)
            results.extend(items)
            log.append(f"cyberleninka={len(items)}")

        # дедупликация по DOI / URL / нормализованному заголовку
        results = _dedupe(results)[: params.max_results]

        sources = [
            Source(
                kind="scholar",
                title=p.title,
                url=p.url,
                doi=p.doi,
                authors=p.authors,
                year=p.year,
                snippet=p.abstract[:200],
                extra={"engine": p.source_engine, "container": p.container},
            )
            for p in results
        ]

        return ToolResult(
            data=ScholarResponse(publications=results),
            sources=sources,
            log=log,
        )


def _dedupe(items: list[Publication]) -> list[Publication]:
    seen: set[str] = set()
    out: list[Publication] = []
    for p in items:
        key = (p.doi or p.url or p.title.lower().strip())[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
