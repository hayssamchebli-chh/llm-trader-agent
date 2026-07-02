#!/usr/bin/env bash
# start.sh — one-command launcher for the LLM Trader Agent dashboard (Mac/Linux)
# Usage:  ./start.sh     (first run sets everything up; later runs just launch)
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "First run — creating virtual environment…"
  python3 -m venv venv
  source venv/bin/activate
  echo "Installing dependencies (this happens once)…"
  pip install --upgrade pip -q
  pip install -r llm_trader/requirements.txt -q
else
  source venv/bin/activate
fi

echo "Launching dashboard — your browser will open at http://localhost:8501"
streamlit run app.py
