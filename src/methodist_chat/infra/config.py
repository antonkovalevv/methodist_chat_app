"""Конфигурация приложения через pydantic-settings + .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Читаются из переменных окружения и .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["openrouter", "mistral", "gigachat"] = "openrouter"

    openrouter_api_key: str = ""
    openrouter_model: str = "mistralai/mistral-7b-instruct:free"

    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    # --- Поиск ---
    crossref_mailto: str = ""
    openalex_mailto: str = ""
    tavily_api_key: str = ""

    # --- Обработка длинных текстов ---
    # Размер чанка в символах (~3 символа ≈ 1 токен). Для моделей с 32k контекста
    # подходит 24000. Для NVIDIA 262k (nvidia/llama-3.1-nemotron-ultra-253b-v1:free)
    # можно ставить 80000–120000 (оставляя запас на системный промпт + ответ).
    llm_chunk_chars: int = 24_000

    # --- Прочее ---
    data_dir: Path = Field(default=Path("./data"))
    log_level: str = "INFO"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Лениво кэшированный аксессор настроек."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
