# DocMind

RAG-сервис для вопросов к PDF-документам. Документы загружаются через API, после чего по ним можно делать запросы на естественном языке — ответ формируется строго на основе содержимого загруженных файлов.

## Стек

| Компонент | Роль |
| --- | --- |
| **FastAPI** | REST API |
| **Qdrant** | Векторная БД |
| **sentence-transformers** `all-MiniLM-L6-v2` | Локальные эмбеддинги, без внешнего API |
| **pypdf** | Парсинг PDF |
| **Claude Sonnet** (Anthropic) | Генерация ответов + LLM-as-a-judge оценка |

## Запуск

Переменные окружения описаны в `.env.example`. Для работы нужен `ANTHROPIC_API_KEY`.

```bash
cp .env.example .env
docker compose up --build
```

Сервис доступен на `http://localhost:8000`, Qdrant — на `http://localhost:6333`.

## API

### `POST /ingest`

Загрузка PDF-файла. Документ парсится, разбивается на чанки и индексируется в Qdrant.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

```json
{ "status": "ok", "filename": "document.pdf", "chunks": 42 }
```

---

### `POST /query`

Запрос к загруженным документам.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "В чём главный вывод?"}'
```

```json
{
  "answer": "Главный вывод заключается в ...",
  "sources": [
    { "filename": "document.pdf", "chunk_index": 3, "score": 0.91 }
  ]
}
```

**Параметры запроса:**

| Поле | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `question` | `string` | — | Вопрос к документам |
| `top_k` | `int` | `5` | Количество релевантных фрагментов из Qdrant |
| `evaluate` | `bool` | `false` | Включить LLM-as-a-judge оценку ответа |

При `evaluate: true` в ответ добавляется поле `eval`:

```json
{
  "answer": "...",
  "sources": [...],
  "eval": {
    "faithfulness": 0.95,
    "relevance": 0.88,
    "completeness": 0.80,
    "verdict": "PASS",
    "reasoning": "Ответ полностью основан на контексте."
  }
}
```

`verdict` принимает значение `PASS` при `faithfulness >= 0.7` и `relevance >= 0.7`.

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

## Архитектура

```text
Ingest:
  PDF → pypdf → текст → чанки (500 слов, перекрытие 50)
      → all-MiniLM-L6-v2 → векторы → Qdrant

Query:
  вопрос → all-MiniLM-L6-v2 → косинусный поиск в Qdrant (top-k)
          → Claude Sonnet → ответ строго по контексту

Eval (опционально):
  вопрос + фрагменты + ответ → Claude Sonnet → JSON с оценками
```

## Локальная разработка

Зависимости устанавливаются через pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Qdrant можно поднять отдельно, API запускается через uvicorn:

```bash
docker compose up qdrant -d
uvicorn app.main:app --reload
```
