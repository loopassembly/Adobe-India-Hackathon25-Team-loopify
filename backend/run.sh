#!/usr/bin/env bash
set -euo pipefail
exec uvicorn server:app --host 0.0.0.0 --port 8080