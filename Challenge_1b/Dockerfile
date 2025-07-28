FROM --platform=linux/amd64 python:3.10-slim AS runtime

# 1. install build deps only for wheels that need it
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python dependencies
COPY Challenge_1b/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Challenge_1a directory (needed for imports)
COPY Challenge_1a ./Challenge_1a

# 4. Copy Challenge_1b processing script
COPY Challenge_1b/process_collection.py .

# 5. Set the entrypoint
ENTRYPOINT ["python", "process_collection.py", \
    "--input_json", "/app/input/challenge1b_input.json", \
    "--pdf_dir", "/app/input/PDFs", \
    "--output_json", "/app/output/challenge1b_output.json"]
