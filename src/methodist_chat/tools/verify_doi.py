"""verify_doi — проверка живости DOI/URL (HEAD/GET)."""

from __future__ import annotations

from typing import Any

from methodist_chat.infra.tools.verify_doi import VerifyDoiTool, VerifyRequest

from .base import ToolOutput


class VerifyDoi:
    name = "verify_doi"
    description = (
        "Проверяет, что DOI или URL действительно открывается (HTTP 200-399). "
        "Используй после scholar_search, чтобы не предлагать методисту "
        "битую ссылку."
    )
    args_schema = {
        "doi": "DOI без префикса (например, '10.1109/ACCESS.2022.3211167'). Опционально.",
        "url": "полный URL. Опционально, если задан doi.",
    }

    def __init__(self) -> None:
        self._impl = VerifyDoiTool()

    def run(self, args: dict[str, Any]) -> ToolOutput:
        doi = args.get("doi") or None
        url = args.get("url") or None
        if not doi and not url:
            return ToolOutput(text="", error="Нужен либо 'doi', либо 'url'")
        try:
            res = self._impl.run(VerifyRequest(doi=doi, url=url))
        except Exception as e:
            return ToolOutput(text="", error=f"verify_doi: {e}")
        d = res.data
        status = "OK" if d.ok else "НЕДОСТУПЕН"
        return ToolOutput(
            text=(
                f"Статус: {status}\n"
                f"Финальный URL: {d.final_url}\n"
                f"HTTP: {d.status_code} {d.reason}"
            )
        )
