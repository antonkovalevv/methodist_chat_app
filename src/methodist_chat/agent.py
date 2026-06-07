"""Простой ReAct-агент для methodist_chat.

Один цикл = один user-ход. Внутри цикла модель в ≤ 4 итерациях либо
вызывает инструменты (Action) и получает результаты (Observation),
либо отдаёт финальный ответ (Final).

Особенности реализации:

* **Текстовый протокол** (а не нативный function-calling). Модель пишет:

      Thought: …
      ACTION: <name>
      ARGS: { …json… }

  или

      Thought: …
      FINAL:
      <markdown>

  Это работает одинаково на всех 3 провайдерах, не зависит от их
  встроенного tool-API и ОЧЕНЬ удобно показывать на защите — комиссия
  видит «сырые» решения модели.

* **Парсер устойчив**: если модель забыла "ACTION:" или прислала
  невалидный JSON, агент возвращает ей мягкое сообщение «верни план
  ещё раз в правильном формате» и даёт ещё одну попытку.

* **Trace** — сервер записывает все шаги (thought + action + obs) в
  объект ``AgentTrace``, который UI показывает «как агент думал».
  В chat-history сохраняется только финальный ответ.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .history import ChatTurn, compress_history
from .llm import ChatMessage, LLMProvider
from .llm.base import TRUNCATION_MARKER, attachment_budget
from .prompts import build_system_prompt, render_document_card
from .tools import ToolOutput, ToolRegistry


_log = logging.getLogger(__name__)


def _short(s: str, n: int = 80) -> str:
    """Аккуратно обрезанная строка для лога — без переводов строк."""
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_args(args: dict) -> str:
    """Сжимает dict аргументов в одну строку для лога."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}={_short(v, 40)!r}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


# Жёсткий потолок числа итераций ReAct-цикла.
DEFAULT_MAX_ITERATIONS = 4

# Сколько раз пытаемся «дочитать» обрезанный финальный ответ.
MAX_CONTINUATIONS = 4

# Минимальный head_preview документа в системном промпте.
HEAD_PREVIEW_MIN_CHARS = 1500
# Максимальный head_preview — даже на 128k моделях слишком большой кусок
# мешает агенту, лучше пусть подгружает через read_attachment.
HEAD_PREVIEW_MAX_CHARS = 8000

# Порог релевантности, ниже которого preflight rag_search НЕ пушит
# результаты в контекст модели. Идея: если в локальной базе нашлось
# что-то слабо связанное с запросом — лучше не подмешивать этот шум,
# а оставить модели возможность самой вызвать rag_search, если она
# действительно считает это полезным. Перекрывается через .env
# (`RAG_PREFLIGHT_MIN_RELEVANCE`).
DEFAULT_PREFLIGHT_MIN_RELEVANCE = 0.45

# Регулярка для парсинга строки «макс. релевантность: 0.78» из ответа
# rag_search.
_RE_MAX_RELEVANCE = re.compile(
    r"макс\.?\s*релевантность\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _max_relevance_in(text: str) -> float | None:
    """Достаёт «макс. релевантность: X.XX» из текста rag_search.

    Возвращает None, если строки нет или число не распарсилось.
    """
    if not text:
        return None
    m = _RE_MAX_RELEVANCE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def _preflight_relevance_threshold() -> float:
    """Текущий порог релевантности для preflight (читает env)."""
    raw = os.getenv("RAG_PREFLIGHT_MIN_RELEVANCE", "")
    if not raw:
        return DEFAULT_PREFLIGHT_MIN_RELEVANCE
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_PREFLIGHT_MIN_RELEVANCE
    if 0.0 <= v <= 1.0:
        return v
    return DEFAULT_PREFLIGHT_MIN_RELEVANCE


# Карта «инструмент → понятное человеку сообщение для статуса». Когда агент
# вызывает инструмент, UI показывает соответствующий статус, чтобы пользователь
# видел, чем модель сейчас занята.
TOOL_PROGRESS_LABELS: dict[str, str] = {
    "rag_search": "🔎 Ищу в базе знаний…",
    "read_attachment": "📄 Читаю прикреплённый файл…",
    "web_search": "🌐 Ищу в интернете…",
    "scholar_search": "📚 Ищу научные статьи…",
    "fetch_url": "🌐 Загружаю веб-страницу…",
    "verify_doi": "🔬 Проверяю DOI…",
}


# Тип callback-а прогресса: принимает короткую строку статуса.
ProgressCallback = Callable[[str], None]


def _format_llm_error(exc: BaseException) -> str:
    """Делает из tenacity.RetryError / httpx-ошибок понятную человеку строку.

    `RetryError[<Future at 0x... state=finished raised RemoteProtocolError>]`
    превращается в `RemoteProtocolError: соединение разорвано шлюзом провайдера`.
    """
    # tenacity.RetryError несёт last_attempt с реальной ошибкой.
    last_attempt = getattr(exc, "last_attempt", None)
    if last_attempt is not None:
        try:
            inner = last_attempt.exception()
        except Exception:
            inner = None
        if inner is not None:
            return _format_llm_error(inner)
    # У RuntimeError, который мы сами поднимаем в провайдерах, message уже
    # человекочитаемый — отдаём как есть.
    msg = str(exc).strip()
    cls = type(exc).__name__
    if not msg or msg == cls:
        return cls
    if cls in ("RuntimeError", "ValueError"):
        return msg
    return f"{cls}: {msg}"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """Один шаг ReAct-цикла."""

    iteration: int
    raw: str = ""                 # сырой ответ модели на этом шаге
    thought: str = ""             # вычлёненный «Thought:»
    action: str | None = None     # имя инструмента (если был)
    args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""         # текст результата инструмента
    final: str | None = None      # финальный markdown-ответ (если был)
    error: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)  # структурированные ссылки


@dataclass
class AgentTrace:
    user_message: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""


# ---------------------------------------------------------------------------
# Парсер ответа модели
# ---------------------------------------------------------------------------


_RE_ACTION = re.compile(r"^\s*ACTION:\s*([\w_-]+)\s*$", re.MULTILINE)
_RE_FINAL = re.compile(r"^\s*FINAL\s*:?\s*$", re.MULTILINE)


def parse_step(raw: str) -> tuple[str | None, dict[str, Any], str | None, str]:
    """Возвращает (action_name, args, final_text, parse_error_or_empty).

    Логика:
      1. Если в тексте есть `FINAL:` — всё ниже берём как финал.
      2. Иначе ищем `ACTION: <name>`; ниже — JSON в `ARGS:` или ```json```.
      3. Если ни того, ни другого — это parse_error.
    """
    if not raw or not raw.strip():
        return None, {}, None, "пустой ответ модели"

    final_match = _RE_FINAL.search(raw)
    if final_match:
        final_text = raw[final_match.end() :].strip()
        # Если случайно ниже FINAL ещё что-то вроде ACTION — игнорируем.
        return None, {}, final_text, ""

    action_match = _RE_ACTION.search(raw)
    if not action_match:
        return None, {}, None, "ни ACTION:, ни FINAL: не найдены"

    action_name = action_match.group(1).strip()
    rest = raw[action_match.end() :]

    # Сначала пробуем явный блок ARGS:
    args = _parse_args(rest)
    if args is None:
        return action_name, {}, None, f"не удалось распарсить ARGS для {action_name}"
    return action_name, args, None, ""


_RE_ARGS_LINE = re.compile(r"ARGS\s*:\s*", re.MULTILINE)
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_args(s: str) -> dict[str, Any] | None:
    s = s or ""

    # Вариант 1: ``` ... ``` блок.
    m = _RE_JSON_FENCE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Вариант 2: после "ARGS:" — первый сбалансированный {...}.
    m = _RE_ARGS_LINE.search(s)
    if m:
        tail = s[m.end() :].strip()
        candidate = _extract_balanced_json(tail)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Вариант 3: вообще первый сбалансированный {...} в тексте.
    candidate = _extract_balanced_json(s)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Если в исходном тексте нет вообще никакого JSON — вернём пустой dict
    # (не None), чтобы не считать это ошибкой парсинга — у инструмента может
    # не быть обязательных аргументов.
    if "{" not in s:
        return {}
    return None


def _extract_balanced_json(s: str) -> str | None:
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def extract_thought(raw: str | None) -> str:
    """Вытаскивает `Thought:` (если есть) для отображения в UI.

    Берём весь блок до ближайшего ACTION/ARGS/FINAL или конца текста —
    Thought на нескольких строках встречается часто, нет смысла резать
    только первую строку.
    """
    if not raw:
        return ""
    m = re.search(
        r"(?ims)^\s*Thought\s*:\s*(.+?)(?=\n\s*(?:ACTION|FINAL|ARGS)\s*:|\Z)",
        raw,
    )
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Главный класс агента
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    temperature_step: float = 0.1
    # Лимит ответа не хранится в агенте — берём из провайдера
    # (llm.max_output_tokens). У GigaChat-Lite свой потолок, у Mistral свой и т.д.
    tool_obs_max_chars: int = 4000


class Agent:
    """ReAct-агент. Один экземпляр на (provider, registry, attachment)."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        registry: ToolRegistry,
        attachment_filename: str = "",
        attachment_text: str = "",
        config: AgentConfig | None = None,
        prompt_profile: str | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.attachment_filename = attachment_filename
        self.attachment_text = attachment_text or ""
        self.cfg = config or AgentConfig()
        # None → подхватит PROMPT_PROFILE из env, иначе full.
        self.prompt_profile = prompt_profile

    # ---- system prompt -----------------------------------------------------

    def _head_preview_chars(self) -> int:
        """Сколько символов документа класть в head_preview системного промпта.

        Берём 30% от безопасного бюджета вложения у текущей модели,
        зажатые между HEAD_PREVIEW_MIN_CHARS и HEAD_PREVIEW_MAX_CHARS.
        """
        budget = attachment_budget(self.llm)
        return max(HEAD_PREVIEW_MIN_CHARS, min(HEAD_PREVIEW_MAX_CHARS, budget // 3))

    def _build_system_prompt(self) -> str:
        head_chars = self._head_preview_chars()
        document_card = render_document_card(
            filename=self.attachment_filename,
            char_count=len(self.attachment_text),
            head_preview=self.attachment_text[:head_chars],
            head_preview_max_chars=head_chars,
        )
        return build_system_prompt(
            tools_block=self.registry.describe_for_prompt(),
            document_card=document_card,
            profile=self.prompt_profile,
        )

    # ---- main entry --------------------------------------------------------

    def run_turn(
        self,
        *,
        user_message: str,
        history: list[ChatTurn] | None = None,
        previous_summary: str = "",
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[AgentTrace, str]:
        """Прогоняет один ход. Возвращает (trace, новый_summary_или_исходный).

        ``progress_cb`` — необязательный коллбэк, в который агент кладёт
        короткие статусные строки на каждом ключевом шаге («Ищу в базе
        знаний…», «Формулирую ответ…»). UI вешает на него обновление
        st.status, чтобы пользователь видел, чем модель сейчас занята.
        """

        def _say(msg: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(msg)
                except Exception:
                    # callback из UI не должен ронять агент.
                    pass

        trace = AgentTrace(user_message=user_message)
        history = history or []

        turn_t0 = time.monotonic()
        _log.info(
            "▶ turn started: user=%r (history=%d, provider=%s)",
            _short(user_message, 100),
            len(history),
            type(self.llm).__name__,
        )

        # 1. Сжимаем историю (rolling summary + recent turns).
        if history:
            _say("🧵 Перечитываю предыдущие сообщения…")
            _log.info("  compress_history: %d turns → …", len(history))
        compressed = compress_history(
            history,
            llm=self.llm,
            previous_summary=previous_summary,
        )
        history_messages = compressed.to_messages()

        # 2. Цикл ReAct.
        running_messages: list[ChatMessage] = list(history_messages) + [
            ChatMessage(role="user", content=user_message),
        ]
        system_prompt = self._build_system_prompt()

        # 2a. Preflight rag_search — выполняем поиск по локальной БД ДО
        #     первого LLM-вызова и кладём результаты в контекст как
        #     синтетический Observation. Это решает проблему «модели
        #     ленятся звать rag_search» — релевантный контекст уже
        #     виден, остаётся только написать FINAL или дёрнуть
        #     дополнительный инструмент.
        preflight_step = self._preflight_rag_search(
            user_message=user_message,
            running_messages=running_messages,
            progress_cb=progress_cb,
        )
        if preflight_step is not None:
            trace.steps.append(preflight_step)

        for iteration in range(1, self.cfg.max_iterations + 1):
            full_messages = (
                [ChatMessage(role="system", content=system_prompt)] + running_messages
            )
            _say(
                "🤔 Анализирую запрос…"
                if iteration == 1
                else "🤔 Думаю, что делать дальше…"
            )
            llm_t0 = time.monotonic()
            _log.info(
                "  iter %d: → LLM call (%d msgs, max_tokens=%d)",
                iteration,
                len(full_messages),
                self.llm.max_output_tokens,
            )
            if _log.isEnabledFor(logging.DEBUG):
                # Полный дамп запроса — включается через LOG_LEVEL=DEBUG.
                for mi, m in enumerate(full_messages):
                    _log.debug(
                        "    req[%d] %s: %s",
                        mi,
                        m.role,
                        _short(m.content, 800),
                    )
            try:
                raw = self.llm.chat(
                    full_messages,
                    temperature=self.cfg.temperature_step,
                    max_tokens=self.llm.max_output_tokens,
                )
            except Exception as e:
                llm_dt = time.monotonic() - llm_t0
                pretty = _format_llm_error(e)
                _log.warning(
                    "  iter %d: ✕ LLM error after %.1fs: %s",
                    iteration,
                    llm_dt,
                    _short(pretty, 120),
                )
                step = AgentStep(iteration=iteration, error=f"LLM error: {pretty}")
                trace.steps.append(step)
                trace.finish_reason = "llm_error"
                trace.final_answer = (
                    f"Произошла ошибка при обращении к модели:\n\n{pretty}\n\n"
                    "Попробуйте ещё раз или смените провайдера в настройках."
                )
                trace.finished = True
                return trace, compressed.summary

            llm_dt = time.monotonic() - llm_t0
            _log.info(
                "  iter %d: ← LLM (%d chars in %.1fs)",
                iteration,
                len(raw or ""),
                llm_dt,
            )
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("    resp: %s", _short(raw or "", 2000))

            step = AgentStep(iteration=iteration, raw=raw, thought=extract_thought(raw))

            # Если модель вернула пустой ответ — НЕ кладём его в историю как
            # ChatMessage(assistant, ""): Mistral валидирует и отбивает 400
            # «Assistant message must have either content or tool_calls».
            # Кладём заглушку с явным напоминанием формата + сразу шлём
            # подсказку «верни план в правильном формате» — это ровно то же,
            # что делает ветка parse_error ниже, но без лишнего шага.
            if not (raw or "").strip():
                _log.warning(
                    "  iter %d: empty LLM response, sending format-reminder",
                    iteration,
                )
                running_messages.append(
                    ChatMessage(
                        role="assistant",
                        content="(пустой ответ модели)",
                    )
                )
                running_messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Ты вернул пустой ответ. Верни ровно ОДИН из двух "
                            "вариантов:\n"
                            "1. Вызов инструмента: 'Thought: ...\\nACTION: <имя>"
                            "\\nARGS: {\"...\": ...}'\n"
                            "2. Финальный ответ: 'Thought: ...\\nFINAL:\\n"
                            "<markdown>'"
                        ),
                    )
                )
                step.error = "empty_response"
                trace.steps.append(step)
                continue
            running_messages.append(ChatMessage(role="assistant", content=raw))

            action_name, args, final_text, parse_error = parse_step(raw)

            if final_text is not None:
                _say("✍️ Формулирую ответ…")
                # Дочитываем финальный ответ, если упёрлись в max_tokens —
                # серверный wrapper дописывает [ОБРЕЗАНО:] маркер, по нему
                # запускаем continuation-цикл. Так лимит модели исчезает
                # с точки зрения пользователя.
                final_text = self._continue_final_if_truncated(
                    final_text=final_text,
                    system_prompt=system_prompt,
                    running_messages=running_messages,
                    step=step,
                    progress_cb=progress_cb,
                )
                step.final = final_text
                trace.steps.append(step)
                trace.final_answer = final_text
                trace.finished = True
                trace.finish_reason = "final"
                trace.sources = self._collect_sources(trace)
                _log.info(
                    "◀ turn done: finish=final, answer=%d chars (total %.1fs)",
                    len(final_text or ""),
                    time.monotonic() - turn_t0,
                )
                return trace, compressed.summary

            if parse_error:
                step.error = parse_error
                trace.steps.append(step)
                # Подсказываем модели, как исправиться, и даём ещё попытку.
                hint = (
                    "Твой предыдущий ответ не удалось распарсить. Верни ровно ОДИН "
                    "из двух вариантов:\n"
                    "1. Вызов инструмента:\n"
                    "   Thought: ...\n   ACTION: <имя>\n   ARGS: {\"...\": ...}\n"
                    "2. Финальный ответ:\n"
                    "   Thought: ...\n   FINAL:\n   <markdown>\n"
                    f"Ошибка парсинга: {parse_error}"
                )
                running_messages.append(ChatMessage(role="user", content=hint))
                continue

            # Это ACTION — выполняем инструмент.
            assert action_name is not None
            step.action = action_name
            step.args = args
            _say(TOOL_PROGRESS_LABELS.get(action_name, f"⚙️ Использую `{action_name}`…"))
            _log.info(
                "  iter %d: tool %s(%s)",
                iteration,
                action_name,
                _short_args(args),
            )
            tool = self.registry.get(action_name)
            if tool is None:
                obs_text = (
                    f"Инструмент '{action_name}' не зарегистрирован. "
                    f"Доступные: {', '.join(self.registry.names())}."
                )
                step.observation = obs_text
                step.error = "unknown_tool"
                _log.warning(
                    "  iter %d: ✕ unknown tool %r", iteration, action_name
                )
            else:
                tool_t0 = time.monotonic()
                try:
                    out = tool.run(args) or ToolOutput(text="")
                except Exception as e:
                    out = ToolOutput(text="", error=f"исключение в инструменте: {e}")
                tool_dt = time.monotonic() - tool_t0
                out.truncate(self.cfg.tool_obs_max_chars)
                if out.error:
                    obs_text = f"[ошибка инструмента] {out.error}"
                    step.error = out.error
                    _log.warning(
                        "  iter %d: ✕ %s error in %.1fs: %s",
                        iteration,
                        action_name,
                        tool_dt,
                        _short(str(out.error), 120),
                    )
                else:
                    obs_text = out.text or "(пустой результат)"
                    _log.info(
                        "  iter %d: ← %s (%d chars, %d sources, %.1fs)",
                        iteration,
                        action_name,
                        len(obs_text),
                        len(out.sources or []),
                        tool_dt,
                    )
                step.observation = obs_text
                # Сохраняем sources в trace step (для UI).
                step.sources = [
                    {
                        "tool": action_name,
                        "title": s.title,
                        "url": s.url,
                        "snippet": s.snippet,
                        "extra": s.extra,
                    }
                    for s in (out.sources or [])
                ]

            trace.steps.append(step)
            running_messages.append(
                ChatMessage(
                    role="user",
                    content=f"Observation (результат {action_name}):\n{obs_text}",
                )
            )

        # Кончились итерации — попросим модель сформировать финал из того,
        # что уже наблюдала.
        _say("✍️ Собираю итоговый ответ из найденного…")
        _log.info("  max_iterations reached — forcing finalise")
        forced = self._force_finalise(
            system_prompt, running_messages, progress_cb=progress_cb
        )
        trace.final_answer = forced
        trace.finished = True
        trace.finish_reason = "max_iterations"
        trace.sources = self._collect_sources(trace)
        _log.info(
            "◀ turn done: finish=max_iterations, answer=%d chars (total %.1fs)",
            len(forced or ""),
            time.monotonic() - turn_t0,
        )
        return trace, compressed.summary

    # ---- helpers -----------------------------------------------------------

    # Реплики, по которым preflight RAG не выполняется — это бытовые
    # сообщения, где RAG ничего полезного не вернёт, а вызов будет
    # стоить времени и токенов.
    _TRIVIAL_PREFIXES = (
        "привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро",
        "спасибо", "благодарю", "ок", "понял", "ясно", "хорошо",
        "продолжи", "ещё", "еще", "дальше",
    )
    _PREFLIGHT_MIN_CHARS = 12

    def _is_trivial_message(self, text: str) -> bool:
        """Простая эвристика — стоит ли вообще делать preflight RAG."""
        s = (text or "").strip().lower()
        if len(s) < self._PREFLIGHT_MIN_CHARS:
            return True
        # Бытовые приветствия/благодарности — пропускаем.
        for p in self._TRIVIAL_PREFIXES:
            if s.startswith(p) and len(s) <= len(p) + 20:
                return True
        return False

    def _preflight_rag_search(
        self,
        *,
        user_message: str,
        running_messages: list[ChatMessage],
        progress_cb: ProgressCallback | None = None,
    ) -> AgentStep | None:
        """Выполняет авто-rag_search по сообщению пользователя ДО LLM.

        Если получены хиты — добавляет в ``running_messages`` пару
        синтетических сообщений (assistant с ACTION/ARGS, потом user с
        Observation) — для модели это выглядит так, будто rag_search
        уже был вызван по её собственной инициативе. Возвращает
        ``AgentStep`` для отображения в trace UI.

        Если RAG-инструмент недоступен, или сообщение тривиально, или
        совпадений нет — возвращает None (preflight пропускается, идём
        в обычный ReAct-цикл).

        Этот метод проектируется быть **полностью отказоустойчивым**:
        любое исключение внутри подавляется и приводит к None. Цель —
        чтобы preflight никогда не ронял основной run_turn.
        """
        try:
            return self._preflight_rag_search_impl(
                user_message=user_message,
                running_messages=running_messages,
                progress_cb=progress_cb,
            )
        except Exception:
            # Preflight — best-effort оптимизация. Любой сбой = silently
            # skip, обычный ReAct-цикл всё равно сможет вызвать rag_search.
            return None

    def _preflight_rag_search_impl(
        self,
        *,
        user_message: str,
        running_messages: list[ChatMessage],
        progress_cb: ProgressCallback | None = None,
    ) -> AgentStep | None:
        if self._is_trivial_message(user_message):
            _log.info("  preflight rag_search: skipped (trivial message)")
            return None
        tool = self.registry.get("rag_search")
        if tool is None:
            _log.info("  preflight rag_search: skipped (tool not registered)")
            return None

        if progress_cb is not None:
            try:
                progress_cb("🔎 Авто-поиск в локальной базе знаний…")
            except Exception:
                pass

        query = user_message.strip()[:300]
        args = {"query": query, "n_results": 6}
        pf_t0 = time.monotonic()
        _log.info("  preflight rag_search: → %r", _short(query, 80))
        try:
            out = tool.run(args) or ToolOutput(text="")
        except Exception as e:
            out = ToolOutput(text="", error=f"preflight rag_search: {e}")

        # Truncate под лимит observation (защитно — на случай, если
        # инструмент вернул что-то странное).
        try:
            out.truncate(self.cfg.tool_obs_max_chars)
        except Exception:
            pass

        pf_dt = time.monotonic() - pf_t0

        # Если в результате нет содержательного текста или это явный пустой
        # ответ — preflight не вносим в контекст (это лишний шум для модели).
        text = (out.text or "").strip()
        if out.error or not text:
            _log.info(
                "  preflight rag_search: skipped (empty/error in %.1fs): %s",
                pf_dt,
                _short(str(out.error or "no text"), 80),
            )
            return None
        # Эвристика «пусто/нет совпадений»:
        lowered = text.lower()
        empty_markers = (
            "нет совпадений",
            "не найдено",
            "база не содержит",
        )
        if any(m in lowered for m in empty_markers):
            _log.info(
                "  preflight rag_search: skipped (no hits in %.1fs)", pf_dt
            )
            return None

        # Релевантность ниже порога — НЕ пушим результаты в контекст.
        # Концепция: preflight — это «бесплатная подсказка»; если она
        # сомнительная по качеству, лучше не нагружать модель шумом.
        # Модель при необходимости сама вызовет rag_search.
        max_rel = _max_relevance_in(text)
        threshold = _preflight_relevance_threshold()
        if max_rel is not None and max_rel < threshold:
            _log.info(
                "  preflight rag_search: skipped (max_rel=%.2f < %.2f, %.1fs)",
                max_rel,
                threshold,
                pf_dt,
            )
            return None
        _log.info(
            "  preflight rag_search: ← injected (max_rel=%s, %d sources, %.1fs)",
            f"{max_rel:.2f}" if max_rel is not None else "n/a",
            len(out.sources or []),
            pf_dt,
        )

        # Синтетический step (iteration=0 — выделяет preflight в UI).
        step = AgentStep(
            iteration=0,
            thought="Автоматическая предзагрузка из локальной базы знаний",
            action="rag_search",
            args=args,
            observation=text,
        )
        step.sources = [
            {
                "tool": "rag_search",
                "title": s.title,
                "url": s.url,
                "snippet": s.snippet,
                "extra": s.extra,
            }
            for s in (out.sources or [])
        ]

        # Кладём в running_messages пару (assistant ACTION → user Observation).
        # Модель воспринимает это как «я уже вызвал rag_search, вот результат».
        running_messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    "Thought: первым шагом сверяюсь с локальной "
                    "RAG-базой по теме запроса.\n"
                    f"ACTION: rag_search\nARGS: {json.dumps(args, ensure_ascii=False)}"
                ),
            )
        )
        running_messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Observation (результат rag_search, авто-предзагрузка):\n"
                    + text
                    + "\n\n[подсказка] Используй эти фрагменты, если они по "
                    "теме. Если их недостаточно — зови scholar_search или "
                    "web_search; повторно звать rag_search с тем же "
                    "запросом не нужно."
                ),
            )
        )
        return step

    def _force_finalise(
        self,
        system_prompt: str,
        running_messages: list[ChatMessage],
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> str:
        running_messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Лимит итераций исчерпан. Сформируй FINAL-ответ методисту, "
                    "опираясь на уже собранные observations. Не зови новые "
                    "инструменты. Используй блок 'FINAL:' и обычный markdown."
                ),
            )
        )
        try:
            raw = self.llm.chat(
                [ChatMessage(role="system", content=system_prompt)] + running_messages,
                temperature=self.cfg.temperature_step,
                max_tokens=self.llm.max_output_tokens,
            )
        except Exception as e:
            return f"Не удалось сформировать финальный ответ ({_format_llm_error(e)})."
        # Если модель снова попыталась вызвать инструмент — отдадим как есть.
        _, _, final_text, _ = parse_step(raw)
        text = (final_text or raw).strip()
        # Перед continuation-вызовом кладём в running_messages именно
        # `text`, а не сырой `raw` — иначе модель увидит свои же маркеры
        # `[ОБРЕЗАНО:]` в истории и решит, что их надо повторить.
        # Mistral 400 защита: пустой content в assistant-message запрещён.
        running_messages.append(
            ChatMessage(role="assistant", content=text or "(пустой ответ модели)")
        )
        text = self._continue_final_if_truncated(
            final_text=text,
            system_prompt=system_prompt,
            running_messages=running_messages,
            step=AgentStep(iteration=0),  # фиктивный step, нам нужен только лог
            progress_cb=progress_cb,
        )
        return text

    def _continue_final_if_truncated(
        self,
        *,
        final_text: str,
        system_prompt: str,
        running_messages: list[ChatMessage],
        step: AgentStep,
        progress_cb: ProgressCallback | None = None,
    ) -> str:
        """Если в ``final_text`` есть маркер [ОБРЕЗАНО:], делает до
        MAX_CONTINUATIONS «дочитывающих» вызовов и склеивает результат.

        После каждой склейки маркер удаляется. Если модель упирается в лимит
        столько раз подряд, последний кусок остаётся как есть с финальной
        пометкой о пределе попыток.
        """
        text = final_text
        continuations_done = 0
        # Базовый контекст диалога фиксируем один раз. На каждой итерации
        # формируем «свежий» continuation-запрос (assistant=head + user=
        # «продолжи»), а не накапливаем их пара-за-парой — иначе на 4-й
        # итерации в контекст уезжает 4×head, что взрывает токен-бюджет
        # и провайдер начинает резать ответ всё сильнее.
        base_messages = list(running_messages)
        for _ in range(MAX_CONTINUATIONS):
            if TRUNCATION_MARKER not in text:
                break
            if progress_cb is not None:
                try:
                    progress_cb(
                        f"📝 Дочитываю длинный ответ "
                        f"(часть {continuations_done + 2})…"
                    )
                except Exception:
                    pass
            _log.info(
                "  continuation: → LLM call (part %d, current %d chars)",
                continuations_done + 2,
                len(text),
            )
            cont_t0 = time.monotonic()
            # Уберём маркер: модель его не должна видеть как часть текста.
            cut_idx = text.find(TRUNCATION_MARKER)
            head = text[:cut_idx].rstrip()
            # Свежий continuation-контекст: base + текущая склейка.
            local_messages = base_messages + [
                ChatMessage(
                    role="assistant",
                    content=f"FINAL:\n{head}",
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "Твой предыдущий финальный ответ был обрезан на лимите "
                        "токенов. Продолжи его РОВНО с того места, где он "
                        "оборвался, не повторяя уже написанное. Не добавляй "
                        "вступление, не пиши «продолжение:», не используй "
                        "FINAL: или ACTION: — просто допиши недостающую часть, "
                        "сохраняя формат (если это таблица — продолжай "
                        "таблицу, если список — продолжай список)."
                    ),
                ),
            ]
            try:
                cont_raw = self.llm.chat(
                    [ChatMessage(role="system", content=system_prompt)]
                    + local_messages,
                    temperature=self.cfg.temperature_step,
                    max_tokens=self.llm.max_output_tokens,
                )
            except Exception as e:
                cont_dt = time.monotonic() - cont_t0
                _log.warning(
                    "  continuation: ✕ failed after %.1fs: %s",
                    cont_dt,
                    _short(_format_llm_error(e), 120),
                )
                # Если continuation сломался — возвращаем то, что было,
                # с пометкой об ошибке вместо маркера.
                return (
                    head
                    + f"\n\n[не удалось дочитать ответ: {_format_llm_error(e)}]"
                )
            cont_dt = time.monotonic() - cont_t0
            _log.info(
                "  continuation: ← LLM (%d chars in %.1fs)",
                len(cont_raw or ""),
                cont_dt,
            )
            cont_text = (cont_raw or "").strip()
            # Если модель снова сунула FINAL: — отрежем префикс.
            cont_text = re.sub(r"^\s*FINAL\s*:?\s*\n?", "", cont_text, count=1)
            text = head + cont_text
            continuations_done += 1
            step.observation = (
                step.observation
                + (f"\n[continuation #{continuations_done}: +{len(cont_text)} символов]")
            ).strip()

        # Если все попытки исчерпаны и маркер всё ещё на месте —
        # уберём маркер и поставим явную пометку.
        if TRUNCATION_MARKER in text:
            cut_idx = text.find(TRUNCATION_MARKER)
            text = (
                text[:cut_idx].rstrip()
                + f"\n\n[не удалось дочитать ответ за {MAX_CONTINUATIONS} попыток — "
                "увеличьте *_MAX_TOKENS или попросите модель кратче]"
            )
        return text

    def _collect_sources(self, trace: AgentTrace) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for step in trace.steps:
            out.extend(step.sources or [])
        return out
