@echo off
echo Starting PDF Chatbot Server...
echo.
cd /d "%~dp0"
python -m uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
pause



