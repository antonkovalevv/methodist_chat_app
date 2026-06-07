"""Проверка живых DOI/URL. Делает HEAD-запрос и возвращает статус."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from methodist_chat.infra.logging import get_logger
from .base import Tool, ToolResult
from .registry import register

logger = get_logger(__name__)


class VerifyRequest(BaseModel):
    doi: str | None = None
    url: str | None = None
    timeout: float = 8.0


class VerifyResponse(BaseModel):
    ok: bool
    final_url: str = ""
    status_code: int = 0
    reason: str = ""


def _check(url: str, timeout: float) -> VerifyResponse:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "methodist-assistant/0.1"}) as c:
            try:
                r = c.head(url)
                if r.status_code >= 400:
                    # некоторые сайты не любят HEAD — пробуем GET
                    r = c.get(url)
            except httpx.HTTPError:
                r = c.get(url)
        ok = 200 <= r.status_code < 400
        return VerifyResponse(
            ok=ok,
            final_url=str(r.url),
            status_code=r.status_code,
            reason=r.reason_phrase or "",
        )
    except Exception as e:  # pragma: no cover
        return VerifyResponse(ok=False, final_url=url, status_code=0, reason=str(e))


@register("verify_doi")
class VerifyDoiTool(Tool[VerifyRequest, VerifyResponse]):
    description = "Проверяет, что DOI/URL действительно открывается (HEAD/GET, 200–399)."

    def run(self, params: VerifyRequest) -> ToolResult[VerifyResponse]:
        url = ""
        if params.doi:
            url = f"https://doi.org/{params.doi.lstrip('https://doi.org/').lstrip('/')}"
        elif params.url:
            url = params.url
        else:
            return ToolResult(
                data=VerifyResponse(ok=False, reason="ни DOI, ни URL не задан"),
                log=["empty input"],
            )
        result = _check(url, params.timeout)
        return ToolResult(data=result, log=[f"{url} -> {result.status_code}"])
