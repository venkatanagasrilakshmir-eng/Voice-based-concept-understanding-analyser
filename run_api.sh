#!/usr/bin/env bash
# Launch the VBCUA FastAPI service layer
set -e
cd "$(dirname "$0")"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
