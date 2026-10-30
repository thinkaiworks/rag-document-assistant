# RAG Document Assistant

**Stop answering the same support question 50 times a week.** Drop in your PDFs, SOPs and FAQs, and your team (or your customers) get accurate answers in seconds, with the exact document and page each answer came from.

Built by [ThinkAIWorks](https://thinkaiworks.com).

## What it does

- **Answers only from your documents.** If the answer isn't in your files, it says so instead of making something up.
- **Cites every answer** with the source file and page number, so people can check it.
- **Hybrid search** (keyword + semantic) finds both exact terms like product codes and loosely worded questions.
- **Upload new documents** from the browser; they're indexed instantly.
- **Works with Claude or OpenAI**, and runs fully offline in extractive mode for demos and testing.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY or OPENAI_API_KEY (optional)
uvicorn app.main:app --reload
```

Open http://localhost:8000 and ask: *"How long does express shipping take?"*

Or with Docker:

```bash
docker build -t rag-assistant . && docker run -p 8000:8000 --env-file .env rag-assistant
```

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/ask` | `{"question": "..."}` → answer, `found`, cited sources |
| `POST` | `/documents` | Upload a PDF / TXT / MD file and re-index |
| `GET` | `/health` | Indexed files and active AI provider |

Example:

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Do you ship to Canada?"}'
```

## How it works

1. **Ingest**: files are split on paragraph boundaries into ~800-character chunks with overlap, so no answer gets cut in half.
2. **Retrieve**: BM25 keyword scoring and OpenAI embeddings (when a key is set) are blended with reciprocal-rank fusion.
3. **Answer**: the top chunks go to Claude or GPT with a strict prompt: use only the sources, cite them, admit when the answer isn't there.

## Tests

```bash
pytest -q
```

## Stack

Python · FastAPI · BM25 + OpenAI embeddings · Claude / OpenAI · Docker
