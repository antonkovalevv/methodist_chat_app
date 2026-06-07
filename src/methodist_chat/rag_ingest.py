"""Ingest для пользовательских документов в RAG-базу.

Что делает:

1. ``chunk_text(text, ...)`` — разбивает текст на пересекающиеся окна по
   умолчанию 600 символов с overlap=100. Это компромисс между
   точностью поиска (короче = точнее) и контекстом для модели (длиннее =
   полезнее). Граница окна по возможности съезжает на ближайший пробел/
   перенос строки.

2. ``ingest_text(text, filename) -> IngestResult`` — основной API:
   * убирает уже существующие чанки этого же файла (replace-семантика),
   * чанкует,
   * добавляет в ту же коллекцию ``constants``, что и seed-данные, но с
     метаданными ``source="user_upload"`` — поиск находит и те, и те;
     UI отличает по `source`.

3. ``list_user_uploads()`` / ``delete_user_upload(filename)`` —
   API для UI: показать пользовательские файлы и удалить по имени.

Лимит размера: 1 МБ исходного текста на файл (≈ 200 страниц A4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from methodist_chat.infra.tools.vector_db import VectorDB


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

USER_SOURCE_TAG = "user_upload"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
MAX_DOC_CHARS = 1_000_000  # 1 МБ — выше — отказ
MIN_CHUNK_CHARS = 50       # совсем хвостики игнорим


# ---------------------------------------------------------------------------
# Чанкинг
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Разбивает text на пересекающиеся окна.

    Жадный алгоритм: окно ``[pos, pos+chunk_size]`` сдвигается на
    ``chunk_size - overlap``. Если граница попала в середину слова —
    отступаем влево к ближайшему пробелу/переводу строки (но не больше,
    чем на 80 символов, чтобы не сломать алгоритм на длинных «кашах»).
    """
    if not text or chunk_size <= 0:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(text)
    if n <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Защита от отрицательных/бессмысленных значений параметров.
    overlap = max(0, overlap)
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        # Подвинуть конец влево к ближайшей хорошей границе.
        if end < n:
            search_floor = max(pos + chunk_size // 2, end - 80)
            for j in range(end, search_floor, -1):
                if text[j - 1] in (" ", "\n", "\t", ".", ";", ":", "!", "?"):
                    end = j
                    break
        piece = text[pos:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= n:
            break
        pos = max(pos + step, end - overlap)
    return chunks


# ---------------------------------------------------------------------------
# Ingest API
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    filename: str
    chunks_added: int
    chunks_replaced: int
    char_count: int
    error: str = ""


def _stable_id(filename: str, idx: int, payload: str) -> str:
    """Детерминированный id для чанка — позволяет переиндексировать без хвостов."""
    h = hashlib.sha1(f"{filename}|{idx}|{payload[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"user::{filename}::{idx}::{h}"


def ingest_text(
    text: str,
    filename: str,
    *,
    db: VectorDB | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> IngestResult:
    """Загружает текст в RAG-базу, помечая его ``source='user_upload'``."""
    fname = (filename or "uploaded.txt").strip()
    if not text or not text.strip():
        return IngestResult(
            filename=fname, chunks_added=0, chunks_replaced=0,
            char_count=0, error="пустой текст",
        )
    if len(text) > MAX_DOC_CHARS:
        return IngestResult(
            filename=fname, chunks_added=0, chunks_replaced=0,
            char_count=len(text),
            error=(
                f"документ слишком большой: {len(text)} симв. > {MAX_DOC_CHARS} "
                f"(≈1 МБ). Разделите файл на части."
            ),
        )

    db = db or VectorDB.get()

    # 1. Удаляем старые чанки этого же файла (replace-семантика).
    replaced = 0
    try:
        existing = db.list_all(
            filter={"$and": [
                {"source": USER_SOURCE_TAG},
                {"upload_filename": fname},
            ]},
            limit=10_000,
        )
        if existing:
            ids = [eid for eid, _, _ in existing]
            replaced = db.delete_by_ids(ids)
    except Exception:
        # Не блокируем ingest, если что-то пошло не так с фильтром Chroma.
        pass

    # 2. Чанкуем + добавляем.
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return IngestResult(
            filename=fname, chunks_added=0, chunks_replaced=replaced,
            char_count=len(text),
            error="после чанкинга не осталось ни одного фрагмента",
        )

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadatas: list[dict[str, Any]] = [
        {
            "source": USER_SOURCE_TAG,
            "type": "user",
            "description": fname,
            "upload_filename": fname,
            "uploaded_at": now_iso,
            "chunk_index": i,
            "chunk_count": len(chunks),
        }
        for i in range(len(chunks))
    ]
    ids = [_stable_id(fname, i, c) for i, c in enumerate(chunks)]
    try:
        added = db.add(texts=chunks, metadatas=metadatas, ids=ids)
    except Exception as e:
        return IngestResult(
            filename=fname, chunks_added=0, chunks_replaced=replaced,
            char_count=len(text), error=f"ошибка добавления в Chroma: {e}",
        )

    return IngestResult(
        filename=fname, chunks_added=added, chunks_replaced=replaced,
        char_count=len(text),
    )


# ---------------------------------------------------------------------------
# Listing / deletion (для UI)
# ---------------------------------------------------------------------------


@dataclass
class UserUpload:
    filename: str
    chunks: int
    uploaded_at: str = ""
    char_count_approx: int = 0


def list_user_uploads(db: VectorDB | None = None) -> list[UserUpload]:
    """Группирует пользовательские чанки по имени файла."""
    db = db or VectorDB.get()
    try:
        rows = db.list_all(filter={"source": USER_SOURCE_TAG}, limit=10_000)
    except Exception:
        return []
    by_file: dict[str, UserUpload] = {}
    for _id, doc, meta in rows:
        name = meta.get("upload_filename") or meta.get("description") or "unknown"
        if name not in by_file:
            by_file[name] = UserUpload(
                filename=name,
                chunks=0,
                uploaded_at=str(meta.get("uploaded_at") or ""),
            )
        by_file[name].chunks += 1
        by_file[name].char_count_approx += len(doc or "")
        # Берём «самую свежую» дату.
        ts = str(meta.get("uploaded_at") or "")
        if ts > by_file[name].uploaded_at:
            by_file[name].uploaded_at = ts
    return sorted(by_file.values(), key=lambda u: (u.uploaded_at, u.filename), reverse=True)


def delete_user_upload(filename: str, db: VectorDB | None = None) -> int:
    """Удаляет все чанки конкретного пользовательского файла. Возвращает число удалённых."""
    if not filename:
        return 0
    db = db or VectorDB.get()
    try:
        rows = db.list_all(
            filter={"$and": [
                {"source": USER_SOURCE_TAG},
                {"upload_filename": filename},
            ]},
            limit=10_000,
        )
    except Exception:
        return 0
    if not rows:
        return 0
    ids = [eid for eid, _, _ in rows]
    return db.delete_by_ids(ids)
