@echo off
title Riel Vision Presentation & Live Demo
echo ==========================================================
echo           RIEL VISION PRESENTATION & LIVE DEMO
echo ==========================================================
echo.
echo Starting local HTTP server at project root...
echo.
echo Launching presentation in your default browser:
echo      http://localhost:8000/submission/v6/RielVision.html
echo.
start "" "http://localhost:8000/submission/v6/RielVision.html"
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo.
python -m http.server 8000
