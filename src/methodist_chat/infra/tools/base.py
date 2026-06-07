"""Базовый класс инструмента и общие pydantic-модели результатов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# ---------- Источники ----------


class Source(BaseModel):
    """Единый формат описания источника, на который агент опирается в ответе."""

    kind: Literal["web", "scholar", "constant", "file", "fgos"] = "web"
    title: str = ""
    url: str | None = None
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    snippet: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    def cite_short(self) -> str:
        """Короткая ссылка для логов/служебного вывода."""
        if self.doi:
            return f"DOI:{self.doi}"
        if self.url:
            return self.url
        return self.title or "<без названия>"


# ---------- Базовый инструмент ----------

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolResult(BaseModel, Generic[OutputT]):
    """Результат вызова любого инструмента: данные + список источников + лог."""

    data: OutputT
    sources: list[Source] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)


class Tool(ABC, Generic[InputT, OutputT]):
    """Базовый класс инструмента агента.

    Каждый инструмент:
      • описывает свой `name` и `description` (для tool-registry и логов);
      • принимает типизированный pydantic-вход;
      • возвращает `ToolResult` с данными и списком источников.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, params: InputT) -> ToolResult[OutputT]:  # pragma: no cover
        ...

    def __call__(self, params: InputT) -> ToolResult[OutputT]:
        return self.run(params)
