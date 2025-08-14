#!/bin/bash
set -euo pipefail

# Inputs (mount a collection under /data)
INPUT_JSON="${INPUT_JSON:-/data/challenge1b_input.json}"
PDF_DIR="${PDF_DIR:-/data/PDFs}"
OUTPUT_JSON="${OUTPUT_JSON:-/data/challenge1b_output_generated.json}"

echo "Using:"
echo "  INPUT_JSON = $INPUT_JSON"
echo "  PDF_DIR    = $PDF_DIR"
echo "  OUTPUT_JSON= $OUTPUT_JSON"

# Optional: if you want to use a trained heading model from 1A, mount it as /app/heading_model.pkl.
# process_file() works fine without it (heuristics fallback), so no hard dependency here.

python challenge1b_runner.py "$INPUT_JSON" "$PDF_DIR" "$OUTPUT_JSON"
echo "Done. Wrote: $OUTPUT_JSON"
