"""Главный Streamlit-чат methodist_chat.

Запуск:

    streamlit run src/methodist_chat/ui/app.py

Что показывает на защите:

* Сайдбар: выбор LLM-провайдера (GigaChat / OpenRouter / Mistral),
  загрузка файла, статус ключей.
* Главное окно: чат с историей. Под каждым ответом ассистента —
  раскрывающийся блок «Как агент думал» (трейс ReAct-цикла) и блок
  «Использованные источники».
"""

from __future__ import annotations

import html as _html
import os
import tempfile
from pathlib import Path

import streamlit as st

# Поднимаем .env, если он лежит рядом со старыми пакетами
import methodist_chat  # noqa: F401  (триггерит load_dotenv внутри __init__)

from methodist_chat.infra.tools.loaders import load_document

from methodist_chat.agent import Agent, AgentTrace
from methodist_chat.history import ChatTurn
from methodist_chat.llm import AVAILABLE_PROVIDERS, make_provider
from methodist_chat.suggestions import (
    detect_doc_kind,
    generate_suggestions,
    suggestions_cache_key,
    BASE_SUGGESTIONS,
)
from methodist_chat.rag_ingest import (
    delete_user_upload,
    ingest_text,
    list_user_uploads,
)
from methodist_chat.tools import build_default_registry


# ---------------------------------------------------------------------------
# Page config + state
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Методист-ассистент",
    page_icon="📚",
    layout="wide",
)


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("provider_id", "gigachat")
    ss.setdefault("provider_error", "")
    ss.setdefault("history", [])  # list[ChatTurn]
    ss.setdefault("rolling_summary", "")
    ss.setdefault("attachment_text", "")
    ss.setdefault("attachment_filename", "")
    ss.setdefault("attachment_chars", 0)
    ss.setdefault("traces", [])  # list[AgentTrace], 1:1 с history по индексу
    # Профиль системного промпта (full | balanced | short). Дефолт — из
    # PROMPT_PROFILE; если не задан — full (максимально надёжный).
    ss.setdefault(
        "prompt_profile",
        (os.getenv("PROMPT_PROFILE") or "full").strip().lower(),
    )
    # Подсказки по загруженному файлу. Формат: list[str] | None.
    # Ключ кэша — (filename, head_hash). При смене файла пересчитываются.
    ss.setdefault("_suggestions", None)  # type: ignore[arg-type]
    ss.setdefault("_suggestions_key", "")
    # Видимость блока подсказок над окном ввода. Всегда свёрнуто
    # по умолчанию — раскрывается только кликом по кнопке
    # «Показать подсказки». Сами подсказки генерируются тихо
    # в фоне (при загрузке файла и после каждого ответа).
    ss.setdefault("_show_suggestions", False)
    # Счётчик-«ключ» file_uploader-а: меняем его, чтобы Streamlit показал
    # пустой виджет «Drag & drop» после успешной загрузки. Сам файл
    # остаётся в attachment_text/filename, отображается в карточке ниже.
    ss.setdefault("_uploader_id", 0)
    ss.setdefault("_rag_uploader_id", 0)
    # Очередь: если пользователь кликнул подсказку, текст ложится сюда
    # и на следующем rerun обрабатывается как обычное сообщение.
    ss.setdefault("_pending_user_message", "")


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _provider_status(pid: str) -> tuple[bool, str]:
    """Возвращает (готов, описание)."""
    if pid == "gigachat":
        ok = bool(os.getenv("GIGACHAT_AUTH_KEY"))
        model = os.getenv("GIGACHAT_MODEL", "GigaChat")
    elif pid == "openrouter":
        ok = bool(os.getenv("OPENROUTER_API_KEY"))
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    elif pid == "mistral":
        ok = bool(os.getenv("MISTRAL_API_KEY"))
        model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    else:
        return False, "неизвестный провайдер"
    return ok, (
        f"`{model}` подключён" if ok else f"нет ключа для `{model}` (см. .env)"
    )


with st.sidebar:
    st.title("📚 Методист-ассистент")

    st.subheader("Модель")
    st.session_state["provider_id"] = st.radio(
        "Провайдер LLM",
        options=list(AVAILABLE_PROVIDERS),
        index=list(AVAILABLE_PROVIDERS).index(st.session_state["provider_id"]),
        key="provider_radio",
        format_func=lambda x: {
            "gigachat": "GigaChat (Сбер)",
            "openrouter": "OpenRouter",
            "mistral": "Mistral",
        }.get(x, x),
        label_visibility="collapsed",
    )
    ok, desc = _provider_status(st.session_state["provider_id"])
    (st.success if ok else st.error)(desc)

    # Профиль системного промпта (full / balanced / short) подхватывается
    # из переменной PROMPT_PROFILE в .env (см. _init_state). Переключателя
    # в UI нет — для релиза по умолчанию full, тонкая настройка через .env.

    st.divider()
    st.subheader("Документ")
    # Лимит размера вложения — бьётся об оба конца:
    # 1) Streamlit `server.maxUploadSize` (в .streamlit/config.toml) — срезает
    #    попытку залива большого файла ещё на этапе передачи;
    # 2) ATTACHMENT_MAX_BYTES (дефолт 25 МБ) — второй барьер на
    #    серверной стороне (вдруг config.toml разрешает больше).
    _ATTACHMENT_MAX_BYTES = int(os.getenv("ATTACHMENT_MAX_BYTES", str(25 * 1024 * 1024)))

    uploaded = st.file_uploader(
        "Прикрепить файл (.pdf, .docx, .txt, .md)",
        # .doc убрал: python-docx его не читает (бинарный Word 97-2003).
        # Лучше вовсе не дать загрузить, чем принять и вывалиться сквозь
        # странную ошибку «Package not found».
        type=["pdf", "docx", "txt", "md"],
        # Меняем key после успешной загрузки — Streamlit пересоздаёт
        # виджет с пустым состоянием. Без этого имя файла остаётся
        # болтаться над кнопкой «Browse files», что путает пользователя.
        key=f"attachment_uploader_{st.session_state['_uploader_id']}",
    )
    if uploaded is not None:
        # Парсим, только если файл изменился (по имени+размеру).
        sig = (uploaded.name, uploaded.size)
        if uploaded.size == 0:
            st.error(
                "Файл пустой (0 байт). Проверьте исходный файл и "
                "попробуйте загрузить ещё раз."
            )
        elif uploaded.size > _ATTACHMENT_MAX_BYTES:
            st.error(
                f"Файл слишком большой: {uploaded.size / 1024 / 1024:.1f} МБ > "
                f"лимит {_ATTACHMENT_MAX_BYTES / 1024 / 1024:.0f} МБ. Разделите файл "
                "на части или поднимите ATTACHMENT_MAX_BYTES."
            )
        elif sig != st.session_state.get("_last_file_sig"):
            with tempfile.TemporaryDirectory() as td:
                # Санитизируем имя, чтобы «../» в uploaded.name не выводило
                # путь из временной папки. Path.name оставляет только basename.
                safe_name = Path(uploaded.name).name or "upload"
                tmp = Path(td) / safe_name
                tmp.write_bytes(uploaded.getvalue())
                try:
                    res = load_document(tmp)
                except Exception as e:
                    st.error(f"Не удалось прочитать файл: {e}")
                else:
                    st.session_state["attachment_text"] = res.text
                    st.session_state["attachment_filename"] = uploaded.name
                    st.session_state["attachment_chars"] = res.char_count
                    st.session_state["_last_file_sig"] = sig
                    # Сбрасываем кэш подсказок и автоматически раскрываем
                    # блок над окном ввода — пользователь сразу видит, что
                    # можно сделать с только что загруженным файлом.
                    st.session_state["_suggestions"] = None
                    st.session_state["_suggestions_key"] = ""
                    # Блок остаётся свёрнутым — подсказки сгенерируются
                    # в фоне при следующем рендере, и пользователь увидит их
                    # только по клику по кнопке «Показать подсказки».
                    # Сбрасываем сам виджет — у него новый key на следующем
                    # rerun (см. передачу _uploader_id в key выше).
                    st.session_state["_uploader_id"] += 1
                    if res.char_count == 0:
                        st.warning(
                            f"Файл {uploaded.name} загружен, но из него не удалось "
                            "извлечь текст (возможно, это скан или PDF без "
                            "текстового слоя). Работа с таким вложением "
                            "ограничена."
                        )
                    else:
                        st.success(
                            f"Загружено: {uploaded.name} ({res.char_count} симв.)"
                        )
                    st.rerun()

    if st.session_state["attachment_filename"]:
        st.caption(
            f"Текущий файл: **{st.session_state['attachment_filename']}** "
            f"({st.session_state['attachment_chars']} симв.)"
        )
        if st.button("Отвязать файл", use_container_width=True):
            st.session_state["attachment_text"] = ""
            st.session_state["attachment_filename"] = ""
            st.session_state["attachment_chars"] = 0
            st.session_state.pop("_last_file_sig", None)
            st.session_state["_suggestions"] = None
            st.session_state["_suggestions_key"] = ""
            st.session_state["_show_suggestions"] = False
            st.rerun()

    st.divider()
    st.subheader("База знаний (RAG)")
    st.caption(
        "Файлы здесь индексируются в локальную векторную БД и доступны "
        "ассистенту через `rag_search` во всех будущих сессиях."
    )
    rag_uploaded = st.file_uploader(
        "Добавить документ в RAG",
        type=["pdf", "docx", "txt", "md"],
        key=f"rag_uploader_{st.session_state['_rag_uploader_id']}",
    )
    if rag_uploaded is not None:
        rag_sig = ("rag", rag_uploaded.name, rag_uploaded.size)
        if rag_uploaded.size == 0:
            st.error("Файл пустой (0 байт).")
        elif rag_uploaded.size > _ATTACHMENT_MAX_BYTES:
            st.error(
                f"Файл слишком большой: {rag_uploaded.size / 1024 / 1024:.1f} МБ > "
                f"лимит {_ATTACHMENT_MAX_BYTES / 1024 / 1024:.0f} МБ. Разделите "
                "файл на части или поднимите ATTACHMENT_MAX_BYTES."
            )
        elif rag_sig != st.session_state.get("_last_rag_sig"):
            with tempfile.TemporaryDirectory() as td:
                safe_rag_name = Path(rag_uploaded.name).name or "upload"
                tmp = Path(td) / safe_rag_name
                tmp.write_bytes(rag_uploaded.getvalue())
                try:
                    parsed = load_document(tmp)
                except Exception as e:
                    st.error(f"Не удалось прочитать файл: {e}")
                    parsed = None
            if parsed is not None:
                if parsed.char_count == 0:
                    st.error(
                        "Из файла не удалось извлечь текст — индексировать нечего."
                    )
                else:
                    with st.spinner(
                        f"Индексирую «{rag_uploaded.name}» "
                        f"({parsed.char_count} симв.)…"
                    ):
                        res = ingest_text(parsed.text, rag_uploaded.name)
                    if res.error:
                        st.error(f"Ошибка ingest: {res.error}")
                        # Не ставим _last_rag_sig — пользователь сможет
                        # повторить попытку без перелогина.
                    else:
                        msg = (
                            f"Добавлено в RAG: «{res.filename}» — "
                            f"{res.chunks_added} чанков"
                        )
                        if res.chunks_replaced:
                            msg += (
                                f" (заменено старых: {res.chunks_replaced})"
                            )
                        st.session_state["_last_rag_sig"] = rag_sig
                        st.success(msg)
                        # Сбрасываем виджет, чтобы он показал «Drag & drop».
                        st.session_state["_rag_uploader_id"] += 1
                        st.rerun()

    try:
        user_uploads = list_user_uploads()
    except Exception as e:
        user_uploads = []
        st.warning(f"RAG-БД недоступна: {e}")
    if user_uploads:
        with st.expander(f"Файлов в RAG: {len(user_uploads)}", expanded=False):
            for u in user_uploads:
                col1, col2 = st.columns([4, 1])
                with col1:
                    when = u.uploaded_at[:16].replace("T", " ") if u.uploaded_at else "—"
                    # XSS-безопасный рендер: имя файла экранируется, вторая
                    # строка отрисовывается через st.caption вместо raw HTML.
                    safe_name = _html.escape(u.filename)
                    st.markdown(f"**{safe_name}**")
                    st.caption(
                        f"{u.chunks} чанков · {u.char_count_approx} симв. · {when}"
                    )
                with col2:
                    if st.button("🗑", key=f"del_rag_{u.filename}"):
                        n = delete_user_upload(u.filename)
                        st.toast(f"Удалено {n} чанков из «{u.filename}»")
                        st.rerun()

    st.divider()
    if st.button("🗑 Очистить чат", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["traces"] = []
        st.session_state["rolling_summary"] = ""
        st.session_state.pop("_cached_llm", None)
        st.session_state.pop("_cached_registry", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


st.markdown("### Чат с ассистентом")
if not st.session_state["history"]:
    st.caption(
        "Загрузите документ слева и задайте вопрос. Ассистент сам решит, "
        "когда обратиться к интернету, RAG-базе или научным источникам."
    )


def _render_sources(sources: list[dict]) -> None:
    """Все источники (RAG + внешние) — в одном свёрнутом блоке.

    Сознательно без выделения блока «ваши файлы из базы знаний»: это
    нагружало страницу и не подходило под концепцию чата. Источники
    остаются доступны для проверки, но по умолчанию свёрнуты.
    """
    if not sources:
        return
    with st.expander(
        f"📚 Использованные источники ({len(sources)})", expanded=False
    ):
        for s in sources:
            extra = s.get("extra") or {}
            origin = extra.get("source") or ""
            title = s.get("title") or "(без названия)"
            url = s.get("url") or ""
            snippet = (s.get("snippet") or "")[:200]
            if origin == "user_upload":
                fname = extra.get("upload_filename") or title
                chunk_idx = extra.get("chunk_index", "?")
                chunk_total = extra.get("chunk_count", "?")
                st.markdown(
                    f"- 📂 **{fname}** · фрагмент {chunk_idx}/{chunk_total}\n"
                    f"  > {snippet}"
                )
            elif url:
                tag = f" · _{origin}_" if origin else ""
                st.markdown(f"- **[{title}]({url})**{tag} — {snippet}")
            else:
                tag = f" · _{origin}_" if origin else ""
                st.markdown(f"- **{title}**{tag} — {snippet}")


# История
for i, turn in enumerate(st.session_state["history"]):
    with st.chat_message("user"):
        st.markdown(turn.user)
    with st.chat_message("assistant"):
        st.markdown(turn.assistant)
        trace: AgentTrace | None = None
        if i < len(st.session_state["traces"]):
            trace = st.session_state["traces"][i]
        if trace and trace.sources:
            _render_sources(trace.sources)
        # Блок «Как ассистент работал над ответом» (трейс ReAct-цикла)
        # убран из UI к релизу — пользователю он не несёт пользы. Сам
        # трейс остаётся в session_state.traces для логов/отладки.


# --- Блок подсказок над окном ввода ---
# Показывается, когда есть загруженный файл ИЛИ хотя бы один ход в
# истории (после ответа модели — follow-up подсказки по контексту).
# Toggle-кнопка и авто-показ при загрузке нового файла / после ответа.
_fname_main = st.session_state["attachment_filename"]
_atext_main = st.session_state["attachment_text"]
_history_main = st.session_state["history"]
_last_user_main = _history_main[-1].user if _history_main else ""
_last_asst_main = _history_main[-1].assistant if _history_main else ""

if _fname_main or _history_main:
    # Кэш-ключ зависит от файла И последнего ответа: при новом ответе
    # модели ключ меняется → подсказки пересчитываются по контексту.
    _skey_main = suggestions_cache_key(
        _fname_main, _atext_main, context=_last_asst_main
    )
    # Регенерируем кэш только при смене ключа (файл сменился или новый
    # ответ модели). Без этого LLM-вызов делался бы на каждый rerender
    # Streamlit'а — слишком дорого.
    if st.session_state.get("_suggestions_key") != _skey_main:
        _kind_main = detect_doc_kind(_fname_main, _atext_main)
        _llm_for_hints = None
        try:
            _llm_for_hints = make_provider(st.session_state["provider_id"])
        except Exception:
            pass
        _is_followup = bool(_last_asst_main)
        _spin = (
            "💡 Обновляю подсказки по ответу…"
            if _is_followup
            else "💡 Готовлю подсказки по файлу…"
        )
        with st.spinner(_spin):
            st.session_state["_suggestions"] = generate_suggestions(
                llm=_llm_for_hints,
                filename=_fname_main,
                head_text=_atext_main,
                kind=_kind_main,
                last_user_message=_last_user_main,
                last_assistant_message=_last_asst_main,
            )
        st.session_state["_suggestions_key"] = _skey_main
        # Автоматически раскрываем блок после ответа модели — методист
        # сразу видит, что предлагается делать дальше.
        # Не раскрываем блок автоматически после ответа модели:
        # подсказки генерируются в фоне, но видымые только по
        # явному клику пользователя по кнопке «Показать подсказки».
        _ = _is_followup  # оставлено для трейса/будущей логики

    _hints_main = st.session_state.get("_suggestions") or []
    _show = bool(st.session_state.get("_show_suggestions"))
    _toggle_label = (
        "🔽 Скрыть подсказки" if _show else "💡 Показать подсказки"
    )
    if st.button(_toggle_label, key="toggle_suggestions"):
        st.session_state["_show_suggestions"] = not _show
        st.rerun()

    if _show and _hints_main:
        if _last_asst_main:
            st.caption(
                "Что ещё можно спросить — клик отправляет запрос в чат:"
            )
        elif _fname_main:
            st.caption(
                f"Подсказки по файлу **{_fname_main}** — клик отправляет запрос в чат:"
            )
        else:
            st.caption("Подсказки — клик отправляет запрос в чат:")
        # Раскладываем подсказки в 2 колонки, чтобы они не «съезжали» вниз
        # на узком экране и оставались на расстоянии вытянутой руки от
        # окна ввода.
        _cols = st.columns(2)
        for _idx, _hint in enumerate(_hints_main):
            with _cols[_idx % 2]:
                if st.button(
                    _hint,
                    key=f"suggest_{_idx}",
                    use_container_width=True,
                ):
                    st.session_state["_pending_user_message"] = _hint
                    # После клика прячем блок, чтобы не загораживал ответ.
                    st.session_state["_show_suggestions"] = False
                    st.rerun()


# Ввод
prompt = st.chat_input("Ваш вопрос методисту-ассистенту…")
# Если пользователь кликнул подсказку — её текст лежит в
# _pending_user_message и обрабатывается как обычный chat_input.
_pending = st.session_state.pop("_pending_user_message", "")
if not prompt and _pending:
    prompt = _pending
if prompt:
    user_text = prompt.strip()
    if not user_text:
        st.stop()

    # Сразу показываем сообщение пользователя
    with st.chat_message("user"):
        st.markdown(user_text)

    # КРИТИЧНО: ниже всё обёрнуто в try/except. Любая необработанная
    # ошибка (сеть, ChromaDB, провайдер LLM) НЕ должна терять вопрос
    # пользователя из истории — иначе на rerun страница покажется
    # пустой («чат пропал»). На любой сбой мы:
    #   1) показываем читаемую ошибку как сообщение ассистента;
    #   2) кладём пару (user, error_text) в history — она останется
    #      на экране и при перезагрузке.

    final_answer: str = ""
    trace: AgentTrace | None = None
    new_summary = st.session_state["rolling_summary"]

    # Кэшируем LLM-провайдера в session_state, чтобы не пересоздавать его
    # на каждый ход (GigaChat каждый раз бьёт OAuth, OpenRouter/Mistral —
    # пересобирают httpx-клиент). Сбрасываем при смене провайдера.
    try:
        cached = st.session_state.get("_cached_llm")
        if cached and getattr(cached, "provider_id", None) == st.session_state["provider_id"]:
            llm = cached
        else:
            llm = make_provider(st.session_state["provider_id"])
            st.session_state["_cached_llm"] = llm
    except Exception as e:
        err_text = (
            f"Не удалось подключить провайдера "
            f"`{st.session_state['provider_id']}`: {e}\n\n"
            "Проверьте `.env` или выберите другого провайдера в сайдбаре."
        )
        with st.chat_message("assistant"):
            st.error(err_text)
        st.session_state["history"].append(
            ChatTurn(user=user_text, assistant=err_text)
        )
        st.session_state["traces"].append(
            AgentTrace(
                user_message=user_text,
                final_answer=err_text,
                finished=True,
                finish_reason="provider_error",
            )
        )
        st.stop()

    from methodist_chat.llm.base import attachment_budget as _attbudget

    try:
        # Кэшируем и registry: пересобираем только при смене провайдера
        # или вложения. Иначе каждый ход плодит тяжёлые объекты
        # (ScholarSearchTool, WebSearchTool, ConstantsSearchTool).
        reg_sig = (
            st.session_state["provider_id"],
            st.session_state.get("attachment_filename", ""),
            st.session_state.get("attachment_chars", 0),
            llm.max_output_tokens,
        )
        cached_reg = st.session_state.get("_cached_registry")
        if cached_reg and st.session_state.get("_cached_registry_sig") == reg_sig:
            registry = cached_reg
        else:
            registry = build_default_registry(
                attachment_text=st.session_state["attachment_text"],
                filename=st.session_state["attachment_filename"],
                attachment_chars_budget=_attbudget(llm),
            )
            st.session_state["_cached_registry"] = registry
            st.session_state["_cached_registry_sig"] = reg_sig
        agent = Agent(
            llm=llm,
            registry=registry,
            attachment_filename=st.session_state["attachment_filename"],
            attachment_text=st.session_state["attachment_text"],
            prompt_profile=st.session_state.get("prompt_profile") or None,
        )

        with st.chat_message("assistant"):
            # Live-статус: пока агент работает, в шапке статуса меняется
            # подпись «Ищу в базе знаний…», «Читаю файл…» и т.д.
            with st.status("🤔 Анализирую запрос…", expanded=False) as status:
                def _on_progress(msg: str) -> None:
                    # Streamlit потокобезопасно перерисовывает status.update.
                    try:
                        status.update(label=msg)
                    except Exception:
                        # Если status уже закрыт/невалиден — не фатально.
                        pass

                trace, new_summary = agent.run_turn(
                    user_message=user_text,
                    history=list(st.session_state["history"]),
                    previous_summary=st.session_state["rolling_summary"],
                    progress_cb=_on_progress,
                )
                status.update(label="Готово", state="complete", expanded=False)

            if trace.final_answer and "не удалось дочитать" in trace.final_answer:
                st.warning(
                    "Модель не смогла дочитать длинный ответ за несколько "
                    "попыток. Попросите более компактный вариант или "
                    "смените модель."
                )
            final_answer = trace.final_answer or "(пустой ответ)"
            st.markdown(final_answer)
            if trace.sources:
                _render_sources(trace.sources)
            # Блок «Как ассистент работал над ответом» убран из UI
            # к релизу — см. рендер истории выше.
    except Exception as e:
        # Любая внутренняя ошибка (рантайм-баг, сеть, что угодно).
        import traceback as _tb

        _tb.print_exc()  # в консоль для разработчика
        final_answer = (
            f"Произошла внутренняя ошибка ассистента:\n\n"
            f"```\n{type(e).__name__}: {e}\n```\n\n"
            "Ваш вопрос сохранён в истории — попробуйте отправить его "
            "снова. Если повторяется — попробуйте сменить провайдера "
            "в сайдбаре."
        )
        # Сообщение-ассистент мы могли НЕ успеть отрендерить (если
        # ошибка случилась до st.markdown). Покажем ошибку отдельно.
        with st.chat_message("assistant"):
            st.error(final_answer)
        if trace is None:
            trace = AgentTrace(
                user_message=user_text,
                final_answer=final_answer,
                finished=True,
                finish_reason="ui_error",
            )

    # Сохраняем ход в любом случае (успех/ошибка) — главное, чтобы
    # вопрос пользователя не потерялся из истории.
    st.session_state["history"].append(
        ChatTurn(user=user_text, assistant=final_answer)
    )
    st.session_state["traces"].append(trace)
    st.session_state["rolling_summary"] = new_summary
    # Принудительный rerun: блок подсказок над окном ввода рендерится
    # ВЫШЕ chat_input и к моменту прихода ответа уже отработал со старым
    # состоянием истории. Без rerun подсказки по новому контексту
    # были бы сгенерированы только после следующего пользовательского
    # действия. С rerun они генерируются сразу (в фоне, блок
    # остаётся свёрнутым) — пользователь, открыв блок, увидит
    # актуальные подсказки по последнему ответу модели.
    # Ответ при этом не теряется — он попадает в history и отрисуется
    # из неё на следующем проходе.
    st.rerun()
