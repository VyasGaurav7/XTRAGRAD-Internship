@echo off
title CIFAKE Deep Learning Forensics Platform
echo ===================================================================
echo   CIFAKE: Image Classification & Explainable Identification of AI
echo   Launching interactive forensic dashboard (Figures 2 - 7)...
echo ===================================================================
python run_app.py
if errorlevel 1 (
    echo.
    echo Launch failed. If dependencies are missing, run:
    echo pip install -r requirements.txt
    pause
)
