#!/usr/bin/env bash
# Launch the VBCUA Streamlit frontend
set -e
cd "$(dirname "$0")"
streamlit run frontend/app.py
