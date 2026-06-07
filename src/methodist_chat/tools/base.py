"""Простой тип ``Tool`` для methodist_chat.

Каждый инструмент:

* имеет ``name`` (строка, по которой агент его зовёт);
* имеет ``description`` (включается в системный промпт — описание для LLM);
* имеет ``args_schema`` (словарь {"параметр": "описание"} — тоже идёт в промпт);
* реализует ``run(args: dict) -> ToolOutput``.

``ToolOutput`` несёт два поля:

* ``text`` — компактное человеко-читаемое представление результата для
  модели. Жёсткое ограничение: 4000 символов; иначе модель захлебнётся.
* ``sources`` — структурированные ссылки/DOI/URL, которые UI отрисует
  отдельным блоком «использовано».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Source:
    title: str
    url: str = ""
    snippet: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    text: str
    sources: list[Source] = field(default_factory=list)
    error: str | None = None

    def truncate(self, max_chars: int = 4000) -> "ToolOutput":
        if len(self.text) > max_chars:
            self.text = self.text[:max_chars] + "\n…[truncated]"
        return self


class Tool(Protocol):
    name: str
    description: str
    args_schema: dict[str, str]

    def run(self, args: dict[str, Any]) -> ToolOutput: ...
