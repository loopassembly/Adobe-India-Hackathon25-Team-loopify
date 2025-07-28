FROM --platform=linux/amd64 python:3.10-slim AS runtime

# 1. install build deps only for wheels that need it
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. copy project (keep image small: only Challenge_1a code & model)
COPY sample_dataset/schema ./sample_dataset/schema
COPY process_pdfs.py .
COPY heading_model.pkl .

ENTRYPOINT ["python", "process_pdfs.py", \
    "--input_dir", "/app/input", \
    "--output_dir", "/app/output", \
    "--model", "heading_model.pkl"]
