"""fetch_url — глубокое чтение веб-страницы (BeautifulSoup)."""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .base import Source, ToolOutput

_USER_AGENT = (
    "Mozilla/5.0 (compatible; MethodistChat/0.1; +https://example.org)"
)


# --- SSRF-защита ----------------------------------------------------------
#
# Без явной фильтрации `fetch_url` позволил бы LLM-агенту прочитать любой
# внутренний адрес: localhost, RFC1918 (10/8, 172.16/12, 192.168/16),
# RFC4193 (fc00::/7), AWS metadata (169.254.169.254). Это классический
# SSRF-вектор. Блокируем по двум осям:
#   1) хостнейм — явный black-list (`localhost`, `*.internal`).
#   2) DNS-резолюция — после resolve проверяем, что КАЖДЫЙ адрес — global.
# Через env `FETCH_URL_ALLOW_PRIVATE=1` защиту можно временно отключить
# (например, для тестов на локальном Wiki). По умолчанию выключено.


_BLOCKED_HOSTS = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})
_BLOCKED_HOST_SUFFIXES = (".internal", ".local", ".localhost", ".lan")


def _is_global_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    # ip.is_global исключает private/loopback/link-local/reserved/multicast.
    return ip.is_global


def _ssrf_check(url: str) -> str | None:
    """Возвращает текст ошибки, если URL ведёт во внутреннюю сеть, иначе None."""
    import os

    if os.getenv("FETCH_URL_ALLOW_PRIVATE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "не удалось распарсить URL"
    if not host:
        return "URL не содержит хоста"
    if host in _BLOCKED_HOSTS:
        return f"запрещённый хост: {host}"
    for suffix in _BLOCKED_HOST_SUFFIXES:
        if host.endswith(suffix):
            return f"запрещённый внутренний домен: {host}"
    # Если host — это IP-литерал, проверяем напрямую.
    try:
        ipaddress.ip_address(host)
        if not _is_global_ip(host):
            return f"приватный/loopback IP запрещён: {host}"
        return None
    except ValueError:
        pass
    # Иначе — резолвим и убеждаемся, что ВСЕ адреса global.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"DNS не разрешил {host}: {e}"
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        addr = sockaddr[0]
        if not _is_global_ip(addr):
            return f"DNS вернул приватный IP {addr} для {host}"
    return None


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def _clean(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


class FetchUrl:
    name = "fetch_url"
    description = (
        "Скачивает страницу по URL и возвращает её основной текст без меню, "
        "скриптов и подвалов. Используй, когда сниппета из web_search мало — "
        "например, чтобы прочитать release notes, спецификацию ФГОС, статью."
    )
    args_schema = {
        "url": "полный URL (https://…)",
        "max_chars": "максимум символов в ответе (по умолчанию 3500)",
    }

    def run(self, args: dict[str, Any]) -> ToolOutput:
        url = (args.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolOutput(text="", error="URL должен начинаться с http(s)://")
        ssrf_err = _ssrf_check(url)
        if ssrf_err:
            return ToolOutput(text="", error=f"fetch_url: {ssrf_err}")
        max_chars = int(args.get("max_chars", 3500))
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=20.0,
                headers={"User-Agent": _USER_AGENT},
            ) as c:
                r = c.get(url)
            if r.status_code >= 400:
                return ToolOutput(text="", error=f"HTTP {r.status_code} при загрузке {url}")
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "xml" not in ctype:
                # текстовые ответы (json/plain) — отдаём как есть
                body = r.text[:max_chars]
                return ToolOutput(
                    text=f"URL: {url}\nContent-Type: {ctype}\n\n{body}",
                    sources=[Source(title=url, url=url)],
                ).truncate(max_chars + 200)
            title, text = _clean(r.text)
            text = text[:max_chars]
            return ToolOutput(
                text=f"URL: {url}\nЗаголовок: {title}\n\n{text}",
                sources=[Source(title=title or url, url=url, snippet=text[:300])],
            ).truncate(max_chars + 200)
        except Exception as e:
            return ToolOutput(text="", error=f"fetch_url: {e}")
