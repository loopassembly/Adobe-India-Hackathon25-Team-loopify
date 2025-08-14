FROM python:3.10-slim

# System deps for pdfplumber/poppler
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY server.py .
COPY process_pdfs.py .

# data dirs
RUN mkdir -p /app/data/uploads /app/data/audio

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
