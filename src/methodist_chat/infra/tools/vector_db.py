"""Векторная БД для констант (стиль, ГОСТ, ФГОС, шаблоны).

Использует ChromaDB persistent + multilingual MiniLM embeddings.
Главный фикс относительно прежней версии: clear_collection теперь
работает с правильным именем коллекции и не оставляет легаси-данных.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from methodist_chat.infra.config import get_settings
from methodist_chat.infra.logging import get_logger
from .base import Source, Tool, ToolResult
from .registry import register

logger = get_logger(__name__)

DEFAULT_COLLECTION = "constants"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# pydantic
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    filter: dict[str, Any] | None = None


class SearchHit(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float = 0.0
    relevance: float = 0.0


class SearchResponse(BaseModel):
    hits: list[SearchHit] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Singleton-обёртка
# ---------------------------------------------------------------------------


class VectorDB:
    _instance: "VectorDB | None" = None

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        settings.ensure_dirs()
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Константы: стиль, ГОСТ, ФГОС, шаблоны"},
        )
        logger.info(f"Загрузка эмбеддингов: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

    @classmethod
    def get(cls) -> "VectorDB":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- write ---

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> int:
        if not texts:
            return 0
        if ids is None:
            ts = datetime.utcnow().timestamp()
            ids = [f"doc_{i}_{ts}" for i in range(len(texts))]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        return len(texts)

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Полностью пересоздаёт коллекцию (работает с правильным именем)."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:  # pragma: no cover
            logger.warning(f"delete_collection: {e}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Константы: стиль, ГОСТ, ФГОС, шаблоны"},
        )

    # --- read ---

    def list_all(
        self,
        filter: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Возвращает список (id, text, metadata) для UI-просмотра/удаления."""
        try:
            raw = self.collection.get(where=filter, limit=limit)
        except Exception as e:
            logger.warning(f"VectorDB.list_all: {e}")
            return []
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        out: list[tuple[str, str, dict[str, Any]]] = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            doc_id = ids[i] if i < len(ids) else f"unknown_{i}"
            out.append((doc_id, doc, meta or {}))
        return out

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            logger.warning(f"VectorDB.delete_by_ids: {e}")
            return 0
        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        emb = self.embedder.encode([query]).tolist()
        raw = self.collection.query(
            query_embeddings=emb, n_results=n_results, where=filter
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for d, m, dist in zip(docs, metas, dists):
            hits.append(
                SearchHit(
                    text=d, metadata=m or {}, distance=dist, relevance=1.0 / (1.0 + dist)
                )
            )
        return hits


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@register("constants_search")
class ConstantsSearchTool(Tool[SearchRequest, SearchResponse]):
    description = "Поиск констант (стиль, ГОСТ, ФГОС, шаблоны) в локальной векторной БД."

    def __init__(self):
        # Лениво: реальный VectorDB поднимается при первом вызове
        self._db: VectorDB | None = None

    @property
    def db(self) -> VectorDB:
        if self._db is None:
            self._db = VectorDB.get()
        return self._db

    def run(self, params: SearchRequest) -> ToolResult[SearchResponse]:
        try:
            hits = self.db.search(params.query, params.n_results, params.filter)
        except Exception as e:
            logger.warning(f"constants_search: БД недоступна или пуста: {e}")
            return ToolResult(
                data=SearchResponse(hits=[]),
                log=[f"VectorDB error: {e}"],
            )
        sources = [
            Source(
                kind="constant",
                title=h.metadata.get("description", h.metadata.get("type", "константа")),
                snippet=h.text[:300],
                extra=h.metadata,
            )
            for h in hits
        ]
        return ToolResult(
            data=SearchResponse(hits=hits),
            sources=sources,
            log=[f"hits={len(hits)} query='{params.query[:60]}'"],
        )
