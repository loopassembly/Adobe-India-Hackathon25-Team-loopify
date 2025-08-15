FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY process_pdfs.py .
COPY run.sh .
COPY sample_dataset/ ./sample_dataset/
COPY train_data/ ./train_data/

ENTRYPOINT ["/bin/bash", "run.sh"]