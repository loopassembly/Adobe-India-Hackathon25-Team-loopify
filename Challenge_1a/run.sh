#!/bin/bash
set -e  # Exit immediately on error

echo "Training model..."
python process_pdfs.py \
    --train_dir /app/train_data \
    --model /app/heading_model.pkl

echo "Processing PDFs..."
python process_pdfs.py \
    --input_dir /app/input \
    --output_dir /app/output \
    --model /app/heading_model.pkl

echo "Processing complete. Output files in /app/output"