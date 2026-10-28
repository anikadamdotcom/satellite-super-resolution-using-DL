@echo off
setlocal
cd /d "%~dp0"
if exist ".venv-cpu\Scripts\python.exe" (
    ".venv-cpu\Scripts\python.exe" app.py
) else (
    python app.py
)
if errorlevel 1 (
    echo Application could not start. Run setup_cpu.bat first and check the error above.
    pause
    exit /b 1
)
