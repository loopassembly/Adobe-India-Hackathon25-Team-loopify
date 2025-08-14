FROM python:3.10-slim

# System deps for pdfplumber
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

# Minimal requirements for 1B
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY process_pdfs.py .
COPY challenge1b_runner.py .
COPY run.sh .

# We won't bake the PDFs into the image; mount them at runtime.
ENTRYPOINT ["/bin/bash", "run.sh"]
