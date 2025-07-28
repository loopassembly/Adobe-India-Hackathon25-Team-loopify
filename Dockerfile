FROM --platform=linux/amd64 python:3.10-slim AS runtime


RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY Challenge_1a/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY Challenge_1a/sample_dataset/schema ./sample_dataset/schema
COPY Challenge_1a/process_pdfs.py .
COPY Challenge_1a/heading_model.pkl .


ENTRYPOINT ["python", "process_pdfs.py", \
    "--input_dir", "/app/input", \
    "--output_dir", "/app/output", \
    "--model", "heading_model.pkl"]
