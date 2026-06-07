"""Реестр инструментов methodist_chat.

Регистрация — простая dict, без декораторов. Это упрощает понимание
архитектуры на защите: «вот 6 инструментов, вот их экземпляры, вот
агент, который их зовёт по имени».
"""

from __future__ import annotations

from .base import Source, Tool, ToolOutput
from .fetch_url import FetchUrl
from .read_attachment import ReadAttachment
from .rag_search import RagSearch
from .scholar_search import ScholarSearch
from .verify_doi import VerifyDoi
from .web_search import WebSearch

__all__ = [
    "Source",
    "Tool",
    "ToolOutput",
    "ToolRegistry",
    "build_default_registry",
]


class ToolRegistry:
    """Просто словарь {name: tool_instance} с парой удобных методов."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def describe_for_prompt(self) -> str:
        """Текстовое описание реестра для системного промпта.

        Формат — для каждого инструмента: имя, описание, аргументы.
        """
        lines: list[str] = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
            if t.args_schema:
                for arg, desc in t.args_schema.items():
                    lines.append(f"    {arg}: {desc}")
        return "\n".join(lines)


def build_default_registry(
    attachment_text: str = "",
    filename: str = "",
    *,
    attachment_chars_budget: int = 6000,
) -> ToolRegistry:
    """Собирает реестр со всеми 6 инструментами.

    ``read_attachment`` создаётся с привязкой к текущему вложению (может быть
    пустым — тогда инструмент в run() вернёт понятную ошибку).
    ``attachment_chars_budget`` подбирается под контекстное окно конкретной
    модели (см. methodist_chat.llm.base.attachment_budget).
    """
    # Порядок важен: модель часто читает список инструментов сверху вниз
    # и подсознательно зовёт первый «подходящий». Поэтому rag_search и
    # read_attachment идут первыми, web_search — последним из поисковых.
    reg = ToolRegistry()
    reg.register(RagSearch())
    reg.register(
        ReadAttachment(
            attachment_text=attachment_text,
            filename=filename,
            chars_budget=attachment_chars_budget,
        )
    )
    reg.register(ScholarSearch())
    reg.register(WebSearch())
    reg.register(FetchUrl())
    reg.register(VerifyDoi())
    return reg
