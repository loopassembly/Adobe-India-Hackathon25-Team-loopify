# Loopify: Unified Backend & Frontend (Docker, Port 8080)

This repository contains the full-stack solution for Loopify, combining a Python FastAPI backend and a React/Vite frontend, both served from a single Docker container on port **8080** using nginx as a reverse proxy.

---

## Features

- **Single Dockerfile**: Runs both backend and frontend on port 8080
- **Backend**: FastAPI (Python) for document processing, LLM, TTS, and REST APIs
- **Frontend**: React + Vite + Tailwind CSS for modern UI
- **nginx**: Serves static frontend and proxies API requests to backend

---

## Project Structure

```
├── backend/                # Python FastAPI backend
│   ├── server.py           # Main API server
│   ├── ...                 # Other backend modules
│   └── requirements.txt    # Python dependencies
├── docdots-frontend/       # React + Vite frontend
│   ├── src/                # Frontend source code
│   └── package.json        # Frontend dependencies
├── Dockerfile              # Unified Dockerfile (root)
├── nginx.conf              # nginx config for proxy/static
├── start.sh                # Entrypoint script
└── README.md               # This file
```

---

## Step-by-Step Setup & Usage

### 1. Build the Docker Image

Make sure Docker is installed. Run:

```bash
docker build -t loopify-app .
```

### 2. Run the Container

Use the following command (replace secrets as needed):

```bash
docker run --rm \
  -p 8080:8080 \
  -e ADOBE_EMBED_API_KEY='87fd9dfa2dd74230aa2b211c5e001c8d' \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_MODEL='gemini-2.5-flash' \
  -e GOOGLE_APPLICATION_CREDENTIALS=/backend/credentials/gcp.json \
  -e TTS_PROVIDER=azure \
  -e AZURE_TTS_KEY='your_azure_tts_key' \
  -e AZURE_TTS_ENDPOINT='your_azure_tts_endpoint' \
  loopify-app
```

### 3. Access the App

- Open your browser and go to: [http://localhost:8080](http://localhost:8080)
- The frontend UI will load. All API requests are proxied to the backend.

---

## How It Works

### Dockerfile (root)

- Installs Python, Node.js, nginx
- Installs backend dependencies
- Builds frontend static files
- Configures nginx to serve frontend and proxy `/api` requests to backend
- Entrypoint: `start.sh` launches both backend and nginx

### start.sh

- Starts FastAPI backend on port 9000
- Starts nginx (serves frontend, proxies API to backend)
- Monitors both processes

### nginx.conf

- Serves static files from `/frontend/dist` on `/`
- Proxies API requests to backend at `http://127.0.0.1:9000`
- All traffic is exposed on port 8080

---

## Development (Local)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 9000
```

### Frontend

```bash
cd docdots-frontend
npm install
npm run dev
```

---

## Environment Variables

Set these when running the Docker container:

- `ADOBE_EMBED_API_KEY` - Adobe Embed API Key (`87fd9dfa2dd74230aa2b211c5e001c8d`)
- `LLM_PROVIDER` - LLM provider (e.g., gemini)
- `GEMINI_MODEL` - Gemini model name
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP credentials
- `TTS_PROVIDER` - TTS provider (e.g., azure)
- `AZURE_TTS_KEY` - Azure TTS API key
- `AZURE_TTS_ENDPOINT` - Azure TTS endpoint

---

## Troubleshooting

- Make sure all environment variables are set correctly
- Check Docker build logs for errors
- Ensure port 8080 is not in use
- For API errors, check backend logs

---

## References

- See `backend/README.md` and `docdots-frontend/README.md` for more details on each part
- Dockerfile, nginx.conf, and start.sh in the root for deployment logic

---

## Backend API Endpoints

All endpoints are served by the FastAPI backend. Swagger UI is available at `/swagger`.

| Endpoint           | Method | Description                                                  |
| ------------------ | ------ | ------------------------------------------------------------ |
| `/health`          | GET    | Health and status check (model, cache, etc.)                 |
| `/status`          | GET    | Embedding/model status only                                  |
| `/tts/test`        | GET    | Test TTS synthesis; returns sample audio                     |
| `/docs`            | GET    | List available PDF documents                                 |
| `/index`           | POST   | Upload and/or (re)index PDFs (multipart or disk)             |
| `/outline`         | GET    | Get outline for a document (requires `document` query param) |
| `/recommendations` | POST   | Get related/recommended sections (semantic search)           |
| `/insights`        | POST   | Get structured insights (markdown) for a selection or page   |
| `/podcast`         | POST   | Generate podcast audio (TTS, script planning)                |
| `/select`          | POST   | Search and get insights for a selection                      |

### Example Request: Recommendations

```json
POST /recommendations
{
  "document": "file01.pdf",
  "page": 0,
  "selection": "What is the main topic?",
  "top_k": 5
}
```

### Example Request: Insights

```json
POST /insights
{
  "document": "file01.pdf",
  "page": 0,
  "selection": "Key findings",
  "top_k": 3
}
```

### Example Request: Podcast

```json
POST /podcast
{
  "document": "file01.pdf",
  "page": 0,
  "selection": "Summary of section",
  "style": "podcast",
  "speakers": 2,
  "duration_min": 3.0,
  "voices": ["alloy", "verse"],
  "format": "audio-48khz-192kbitrate-mono-mp3"
}
```

---

## License

This project is licensed under the MIT License.
