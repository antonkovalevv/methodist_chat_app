"""``methodist_chat`` — простой диалоговый ИИ-ассистент для актуализации УММ.

Минимально-сложный прототип для защиты ВКР. Главные принципы:

* **Один LLM-вызов на ход** (или 2-3 при использовании tool-ов через ReAct).
* **3 провайдера** через единый интерфейс: GigaChat, OpenRouter, Mistral.
* **6 базовых инструментов**: web_search, scholar_search, rag_search,
  read_attachment, fetch_url, verify_doi. Никаких compound-инструментов.
* **Оптимизация контекста**: rolling summary истории, drop промежуточных
  tool-результатов, жёсткий token-budget.

Запуск UI:

    streamlit run src/methodist_chat/ui/app.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Грузим .env один раз при импорте пакета — pydantic-settings из старой
# системы тоже читает .env, но не кладёт переменные в os.environ, что
# ломает наши собственные read_env-проверки в провайдерах.
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[2]  # repo root
    for candidate in (_root / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
except Exception:  # pragma: no cover
    pass


# Глушим шумные логгеры зависимостей. Без этого консоль пользователя
# завалена:
#   • стектрейсами `ModuleNotFoundError: torchvision` от Streamlit-watcher,
#     который наугад тыкает каждый submodule пакета `transformers`;
#   • INFO-логами huggingface_hub про каждый HTTP HEAD/GET к модели
#     эмбеддингов (десятки строк на каждый запуск);
#   • предупреждениями transformers про deprecated `__path__`.
# Все они не несут смысла для пользователя — это внутренний шум.
for _name in (
    "streamlit.watcher.local_sources_watcher",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "transformers",
    "transformers.modeling_utils",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "httpx",
    "urllib3",
    "chromadb",
    "chromadb.telemetry",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Дополнительно гасим transformers через их собственный API, если он есть
# (внутренний flag для предупреждений, не покрываемых стандартным logging).
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Telemetry от ChromaDB — она тоже иногда ругается в консоль.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


# А вот наши собственные логи (`methodist_chat.*`) — НАОБОРОТ, делаем
# видимыми. Иначе пользователь не понимает, тупит ли программа или
# просто LLM долго думает. Уровень регулируется через .env переменную
# `LOG_LEVEL` (значения: DEBUG, INFO, WARNING, ERROR; по умолчанию INFO).
_mc_logger = logging.getLogger("methodist_chat")
_mc_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_mc_level = getattr(logging, _mc_level_name, logging.INFO)
_mc_logger.setLevel(_mc_level)
if not any(
    getattr(h, "_methodist_chat_handler", False) for h in _mc_logger.handlers
):
    # Свой консольный handler — чтобы формат был аккуратный, и чтобы
    # сообщения видел сам пользователь в stderr Streamlit'а. Помечаем
    # handler флагом, чтобы при перезагрузке модуля не плодить дубликаты.
    _h = logging.StreamHandler()
    _h._methodist_chat_handler = True  # type: ignore[attr-defined]
    _h.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    _h.setLevel(_mc_level)
    _mc_logger.addHandler(_h)
    # Не пропагируем в root: иначе, если какая-нибудь зависимость
    # (chromadb, sentence_transformers, rich) уже добавила свой handler
    # на root — наши строки появятся в консоли дважды.
    # Тесты получают доступ через tests/conftest.py (он бридж'ит
    # caplog.handler сюда).
    _mc_logger.propagate = False


__version__ = "0.1.0"
