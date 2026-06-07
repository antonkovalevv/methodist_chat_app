"""Реестр инструментов агента. Сценарии работают через `get_tool(name)`."""

from __future__ import annotations

from typing import Callable

from .base import Tool

_factories: dict[str, Callable[[], Tool]] = {}
_instances: dict[str, Tool] = {}


def register(name: str):
    """Декоратор для класса-инструмента: регистрирует фабрику."""

    def _decorate(cls):
        _factories[name] = cls
        cls.name = name
        return cls

    return _decorate


def get_tool(name: str) -> Tool:
    if name in _instances:
        return _instances[name]
    if name not in _factories:
        raise KeyError(f"Инструмент '{name}' не зарегистрирован. Доступны: {list(_factories)}")
    inst = _factories[name]()
    _instances[name] = inst
    return inst


def list_tools() -> list[str]:
    return sorted(_factories)


def clear_instances() -> None:
    """Сбросить кэш инстансов (полезно в тестах)."""
    _instances.clear()
