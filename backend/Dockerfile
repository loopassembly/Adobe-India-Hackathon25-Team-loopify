# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (basic) — slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl tini && \
    rm -rf /var/lib/apt/lists/*

# Copy & install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY process_pdfs.py server.py retrieval.py llm_bridge.py audio_bridge.py /app/
# Adobe-provided scripts (as-is)
COPY chat_with_llm.py generate_audio.py /app/
# Runtime dirs
RUN mkdir -p /app/data/pdfs /app/data/audio /app/data/index

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8080"]