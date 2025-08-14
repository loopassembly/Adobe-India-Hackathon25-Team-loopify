#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
