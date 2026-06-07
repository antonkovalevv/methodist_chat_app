# Methodist Chat

ИИ-ассистент методиста: чат с ReAct-агентом, локальная база знаний (RAG), веб- и научный поиск, загрузка PDF/DOCX.

**Провайдеры LLM:** GigaChat · OpenRouter · Mistral

## Установка

```bash
git clone https://github.com/antonkovalevv/methodist_chat_app.git
cd methodist_chat_app

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -e .
cp .env.example .env   # Windows: copy .env.example .env
```

В `.env` укажите **хотя бы один** API-ключ LLM и почты для научного поиска:

```env
GIGACHAT_AUTH_KEY=          # или OPENROUTER_API_KEY / MISTRAL_API_KEY
CROSSREF_MAILTO=you@example.com
OPENALEX_MAILTO=you@example.com
```

## Запуск

```bash
streamlit run src/methodist_chat/ui/app.py
```

Откройте http://localhost:8501

## База знаний (RAG)

При первом запуске база **пустая**. Загружайте свои документы через сайдбар → **«База знаний (RAG)»** (PDF, DOCX, TXT). Данные хранятся локально в `data/chroma_db/`.

## Требования

- Python 3.10+
- ~2–4 ГБ на диске (зависимости + модель эмбеддингов при первом использовании RAG)
- Интернет (API LLM, веб/научный поиск)

## Структура

```
methodist_chat_app/
├── src/methodist_chat/
│   ├── agent.py, prompts.py, history.py
│   ├── ui/app.py              # Streamlit
│   ├── llm/                   # провайдеры LLM
│   ├── tools/                 # инструменты агента
│   ├── infra/                 # ChromaDB, парсеры, поиск
│   └── scripts/verify_rag.py
├── .streamlit/config.toml
├── .env.example
└── pyproject.toml
```