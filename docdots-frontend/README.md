# Loopify Frontend (React + Vite)

This frontend provides the user interface for Loopify, enabling conversational PDF features and seamless interaction with backend AI services.

---

## Features

- Conversational PDF: chat with documents using natural language
- Modern UI for document upload, retrieval, and interaction
- Real-time semantic search and recommendations
- Insights and podcast/audio generation from document content
- Communicates with backend via REST APIs
- Built with React, Vite, and Tailwind CSS

---

## Note on Performance

Some advanced features, such as conversational PDF analysis and multi-turn queries, may take longer to generate due to the complexity of conversational AI and document understanding. Please allow extra time for these operations.

---

## Step-by-Step Setup

1. **Install Node.js dependencies**
   ```bash
   cd docdots-frontend
   npm install
   ```
2. **Run the frontend locally**
   ```bash
   npm run dev
   ```
3. **Build for production**
   ```bash
   npm run build
   ```
   The build output will be in `dist/` and is served by nginx in Docker.

---

## Main Files

- `src/App.tsx` — Main app component
- `src/api.ts` — API calls to backend
- `src/main.tsx` — Entry point
- `src/styles.css` — Styles
- `tailwind.config.js`, `postcss.config.js` — Tailwind setup

---

## Usage

- Access the app at [http://localhost:8080](http://localhost:8080) when running in Docker
- All API requests are proxied to the backend

---

For more details, see the code in the `src/` directory.