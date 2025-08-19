# Loopify Backend (FastAPI)

This backend powers the Loopify application, providing advanced conversational PDF features, REST APIs for document processing, semantic search, LLM integration, TTS audio generation, and more.

---

## Features

- Conversational PDF: interact with documents using natural language queries
- PDF upload, indexing, and outline extraction
- Semantic search and recommendations
- LLM-powered insights and re-ranking
- Podcast/audio generation with TTS
- Static serving of PDFs and audio
- CORS and range support for PDFs

---

## Note on Performance

Some operations, such as conversational PDF analysis and multi-turn document queries, may take longer to generate due to the complexity of conversational AI and deep document understanding. Please allow extra time for these advanced features.

---

## Step-by-Step Setup

1. **Install Python dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Run the backend server locally**
   ```bash
   uvicorn server:app --reload --port 9000
   ```
3. **Environment Variables**
   Set these for LLM and TTS integration:
   - `LLM_PROVIDER` (e.g., gemini)
   - `GEMINI_MODEL` (e.g., gemini-2.5-flash)
   - `GOOGLE_APPLICATION_CREDENTIALS` (path to GCP credentials)
   - `TTS_PROVIDER` (e.g., azure)
   - `AZURE_TTS_KEY` (Azure TTS API key)
   - `AZURE_TTS_ENDPOINT` (Azure TTS endpoint)

---

## Main Files

- `server.py` — Main FastAPI app and all endpoints
- `process_pdfs.py` — PDF outline extraction
- `retrieval.py` — Semantic search
- `audio_bridge.py` — TTS audio generation
- `chat_with_llm.py` — LLM integration
- `scoring.py` — Scoring utilities

---

## API Endpoints

| Endpoint           | Method | Description                             |
| ------------------ | ------ | --------------------------------------- |
| `/health`          | GET    | Health and status check                 |
| `/status`          | GET    | Embedding/model status                  |
| `/tts/test`        | GET    | Test TTS synthesis                      |
| `/docs`            | GET    | List available PDFs                     |
| `/index`           | POST   | Upload and/or (re)index PDFs            |
| `/outline`         | GET    | Get outline for a document              |
| `/recommendations` | POST   | Get related/recommended sections        |
| `/insights`        | POST   | Get structured insights                 |
| `/podcast`         | POST   | Generate podcast audio                  |
| `/select`          | POST   | Search and get insights for a selection |

---

## Example: Run in Docker

The backend is run as part of the unified Docker container (see root Dockerfile). It is started by `start.sh` and proxied by nginx on port 8080.

---

For more details, see the code in `server.py` and related modules.
