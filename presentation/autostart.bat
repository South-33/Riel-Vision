@echo off
title Riel Vision Presentation & Live Demo Server
echo ==========================================================
echo           RIEL VISION PRESENTATION & LIVE DEMO SERVER
echo ==========================================================
echo.
:: Navigate to project root so all relative paths (../../) work
cd /d "%~dp0\..\.."

echo Starting local HTTP server at project root...
echo.
echo Launching presentation in your default browser:
echo      http://127.0.0.1:8000/submission/v6/RielVision.html
echo.
start "" "http://127.0.0.1:8000/submission/v6/RielVision.html"
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo.
echo ==========================================================
echo.
python -m http.server 8000 --bind 127.0.0.1
