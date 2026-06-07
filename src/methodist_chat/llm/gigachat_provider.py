"""GigaChat-провайдер (Сбер) с тем же интерфейсом, что и ``LLMProvider``.

Зачем нужен:
    Старый ``methodist_assistant.core.llm.LLMProvider`` использует
    OpenAI-совместимый клиент и поэтому работает с OpenRouter / Mistral /
    OpenAI / любой OpenAI-API-compatible моделью. GigaChat (Сбер) тоже даёт
    OpenAI-совместимый ``/chat/completions``, но авторизация у него своя
    (OAuth 2.0 client_credentials), а корневой сертификат сервера —
    «Минцифры РФ», которого нет в стандартном certifi-бандле. Поэтому
    проще написать тонкий собственный клиент, чем хакать OpenAI SDK.

Что реализовано:
    1. Получение access_token через POST ``/api/v2/oauth`` с
       ``Authorization: Basic <auth_key>`` и ``RqUID`` UUID.
    2. Кеш токена с авто-рефрешем за 60 секунд до истечения.
    3. POST ``/api/v1/chat/completions`` с ``Authorization: Bearer <token>``
       в OpenAI-совместимом формате ``messages``.
    4. tenacity-ретраи (3 попытки с экспоненциальным backoff).
    5. По умолчанию ``verify=False`` (проще для разработчика). Включить
       проверку сертификата можно через ``GIGACHAT_VERIFY_SSL=1`` после
       установки CA-бандла Минцифры (см. README_AGENTIC.md).

Совместимость с ``LLMProvider``:
    - конструктор принимает имя модели и ключ;
    - метод ``chat(messages, temperature, max_tokens)`` возвращает строку;
    - метод ``quick(system, user, **kw)`` — сахар для одноразового вызова.

Документация GigaChat:
    https://developers.sber.ru/docs/ru/gigachat/api/overview
    https://developers.sber.ru/docs/ru/gigachat/api/integration
"""

from __future__ import annotations

import time
import uuid
import warnings
from typing import Iterable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from methodist_chat.llm.base import ChatMessage
from methodist_chat.infra.logging import get_logger

logger = get_logger(__name__)


# Эндпоинты GigaChat (актуальные на момент 2025 г.).
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


class GigaChatProvider:
    """OpenAI-совместимый адаптер для GigaChat API.

    Параметры:
        auth_key: уже base64-кодированная строка ``client_id:client_secret``
            из кабинета GigaChat (кнопка «Скопировать ключ»). Хранится в
            ``GIGACHAT_AUTH_KEY``.
        scope: ``GIGACHAT_API_PERS`` (физлицо) или ``GIGACHAT_API_B2B``
            (юрлицо). По умолчанию — ``GIGACHAT_API_PERS``.
        model: имя модели — ``GigaChat``, ``GigaChat-Pro``, ``GigaChat-Max``,
            ``GigaChat-2``, ``GigaChat-2-Pro``, ``GigaChat-2-Max``.
        verify_ssl: проверять ли SSL-сертификат сервера. По умолчанию
            ``False``, потому что у Сбера CA «Минцифры», которого нет в
            стандартном certifi. Если установлен корневой бандл — можно
            поставить ``True``.
        timeout: таймаут на одиночный HTTP-запрос (секунды).
    """

    def __init__(
        self,
        auth_key: str,
        *,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat-2-Max",
        verify_ssl: bool = False,
        timeout: float = 60.0,
    ) -> None:
        if not auth_key:
            raise ValueError(
                "GIGACHAT_AUTH_KEY не задан. Получите Authorization key в "
                "кабинете https://developers.sber.ru/portal/products/gigachat-api "
                "(кнопка «Скопировать ключ») и пропишите в .env."
            )
        self.auth_key = auth_key.strip()
        self.scope = (scope or "GIGACHAT_API_PERS").strip()
        normalized_model = (model or "GigaChat-2-Max").strip()
        # Кабинет Сбера иногда показывает имя «GigaChat-Lite», но в API такой
        # модели нет — базовый/Lite-вариант там называется просто «GigaChat».
        # Заодно нормализуем регистр (Сбер чувствителен).
        if normalized_model.lower().replace("_", "-") in {
            "gigachat-lite",
            "gigachatlite",
            "gigachat lite",
        }:
            logger.info(
                "GigaChat: модель GigaChat-Lite в API называется просто "
                "«GigaChat», заменяю автоматически."
            )
            normalized_model = "GigaChat"
        self.model = normalized_model
        self.verify_ssl = bool(verify_ssl)
        self.timeout = float(timeout)

        if not self.verify_ssl:
            # Один раз скажем «знаем что делаем» — без захламления warnings.
            warnings.filterwarnings(
                "ignore",
                message="Unverified HTTPS request",
                category=Warning,
            )
            logger.info(
                "GigaChat: SSL-проверка отключена (GIGACHAT_VERIFY_SSL=0). "
                "Для production установите CA-бандл Минцифры и поставьте =1."
            )

        self._access_token: str | None = None
        self._expires_at: float = 0.0
        logger.info(
            f"GigaChat готов: model={self.model}, scope={self.scope}, "
            f"verify_ssl={self.verify_ssl}"
        )

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _refresh_token(self) -> str:
        rq_uid = str(uuid.uuid4())
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout) as c:
            r = c.post(
                GIGACHAT_OAUTH_URL,
                headers={
                    "Authorization": f"Basic {self.auth_key}",
                    "RqUID": rq_uid,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self.scope},
            )
        if r.status_code != 200:
            # Скрываем сам ключ из лога, оставляем только тело ответа.
            raise RuntimeError(
                f"GigaChat OAuth неуспешен: HTTP {r.status_code}, тело: "
                f"{r.text[:500]}"
            )
        body = r.json()
        token = body.get("access_token")
        if not token:
            raise RuntimeError(
                f"GigaChat OAuth: в ответе нет access_token. Тело: {body}"
            )
        # expires_at у Сбера приходит как Unix-timestamp в МИЛЛИСЕКУНДАХ.
        # Если поля нет — считаем, что токен живёт 1500 сек (по докам — 30 мин).
        expires_at_ms = body.get("expires_at")
        if isinstance(expires_at_ms, (int, float)) and expires_at_ms > 1e10:
            self._expires_at = float(expires_at_ms) / 1000.0
        else:
            self._expires_at = time.time() + 1500.0
        self._access_token = token
        logger.info(
            f"GigaChat OAuth ok: token обновлён, истечёт через "
            f"{int(self._expires_at - time.time())} с."
        )
        return token

    def _get_token(self) -> str:
        if not self._access_token or time.time() > self._expires_at - 60:
            return self._refresh_token()
        return self._access_token

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: Iterable[ChatMessage] | list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        prepared: list[dict] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                prepared.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                prepared.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                raise TypeError(f"Неподдерживаемый тип сообщения: {type(m)}")

        token = self._get_token()
        # ANSWER-стадия может слать большой запрос; даём чуть больше таймаут
        # на собственно chat-запрос, чем на oauth.
        with httpx.Client(verify=self.verify_ssl, timeout=self.timeout * 2) as c:
            r = c.post(
                GIGACHAT_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Request-ID": str(uuid.uuid4()),
                },
                json={
                    "model": self.model,
                    "messages": prepared,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        if r.status_code == 401:
            # Токен мог истечь раньше времени — пробуем один раз перезапросить.
            logger.info("GigaChat: HTTP 401, обновляю токен и повторяю")
            self._access_token = None
            token = self._get_token()
            with httpx.Client(verify=self.verify_ssl, timeout=self.timeout * 2) as c:
                r = c.post(
                    GIGACHAT_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-Request-ID": str(uuid.uuid4()),
                    },
                    json={
                        "model": self.model,
                        "messages": prepared,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
        if r.status_code != 200:
            raise RuntimeError(
                f"GigaChat chat-запрос неуспешен: HTTP {r.status_code}, "
                f"тело: {r.text[:600]}"
            )
        body = r.json()
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"GigaChat: непонятный ответ chat-API: {body}"
            ) from e

    def quick(self, system: str, user: str, **kw) -> str:
        return self.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            **kw,
        )
