@echo off
REM start.bat — one-command launcher for the LLM Trader Agent dashboard (Windows)
REM Double-click this file, or run it from Command Prompt.
cd /d "%~dp0"

if not exist "venv\" (
    echo First run - creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies (this happens once)...
    python -m pip install --upgrade pip -q
    pip install -r llm_trader\requirements.txt -q
) else (
    call venv\Scripts\activate.bat
)

echo Launching dashboard - your browser will open at http://localhost:8501
streamlit run app.py
pause
