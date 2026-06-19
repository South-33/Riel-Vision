@echo off
title Riel Vision Presentation & Live Demo Server
echo ==========================================================
echo           RIEL VISION PRESENTATION & LIVE DEMO SERVER
echo ==========================================================
echo.
:: Navigate to the directory containing this batch file (v6 folder)
cd /d "%~dp0"

echo Starting local HTTP server inside the v6 directory...
echo.
echo Launching presentation in your default browser:
echo      http://127.0.0.1:8000/Presentation/RielVision.html
echo.
start "" "http://127.0.0.1:8000/Presentation/RielVision.html"
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo.
echo ==========================================================
echo.
python -m http.server 8000 --bind 127.0.0.1
