"""Маленькая утилита: показать, что лежит в RAG, и попробовать запрос.

Использование:

    python -m methodist_chat.scripts.verify_rag                  # список файлов
    python -m methodist_chat.scripts.verify_rag "ваш запрос"     # + поиск

Что выводится:

  • Список загруженных пользователем файлов (имя, число чанков, размер, дата).
  • Если передан запрос — показывает топ-K результатов с релевантностью
    и обрезанным сниппетом.
"""

from __future__ import annotations

import sys
from typing import Iterable

from methodist_chat.rag_ingest import list_user_uploads
from methodist_chat.tools.rag_search import RagSearch


def _print_user_uploads() -> None:
    uploads = list_user_uploads()
    print(f"\n=== Пользовательские документы в RAG: {len(uploads)} ===")
    if not uploads:
        print("(пусто) Загрузите файл через сайдбар Streamlit или ingest_text()")
        return
    for u in uploads:
        when = (u.uploaded_at or "")[:16].replace("T", " ")
        print(
            f"  • {u.filename}  "
            f"({u.chunks} чанков, {u.char_count_approx} симв., {when})"
        )


def _print_search_results(query: str) -> None:
    print(f"\n=== rag_search: «{query}» ===")
    rag = RagSearch()
    out = rag.run({"query": query, "n_results": 5})
    if out.error:
        print(f"  ERROR: {out.error}")
        return
    if not out.sources:
        print("  Никаких чанков не найдено.")
        return
    for i, s in enumerate(out.sources, 1):
        extra = s.extra or {}
        fname = extra.get("upload_filename") or extra.get("description") or "—"
        print(f"\n  [{i}] {fname}")
        snippet = (s.snippet or "").replace("\n", " ")[:300]
        print(f"      {snippet}…")


def main(argv: Iterable[str]) -> int:
    args = list(argv)[1:]
    _print_user_uploads()
    if args:
        query = " ".join(args).strip()
        if query:
            _print_search_results(query)
    else:
        print(
            "\nЗапустите ещё раз с запросом, чтобы увидеть, какие чанки "
            "находит RAG, например:\n"
            "    python -m methodist_chat.scripts.verify_rag \"ФГОС 09.03.03 объём программы\""
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
