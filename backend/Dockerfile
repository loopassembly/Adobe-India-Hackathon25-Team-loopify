# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (basic) — slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl tini nodejs npm nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy credentials
COPY backend/credentials /backend/credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/backend/credentials/gcp.json

# Copy backend code
COPY backend/process_pdfs.py backend/server.py backend/retrieval.py backend/llm_bridge.py backend/audio_bridge.py /app/
COPY backend/chat_with_llm.py backend/generate_audio.py /app/
COPY backend/scoring.py /app/
COPY backend/run.sh /app/run.sh
RUN mkdir -p /app/data/pdfs /app/data/audio /app/data/index

# Copy frontend code
COPY docdots-frontend /frontend
WORKDIR /frontend
RUN npm install && npm run build

# Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 8080

WORKDIR /app

# Entrypoint script to run both backend and nginx
COPY start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/start.sh"]