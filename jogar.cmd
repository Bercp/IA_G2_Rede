@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Execute primeiro: python scripts\install.py
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import importlib.util,sys; sys.exit(any(importlib.util.find_spec(m) is None for m in ('chess','numpy','torch')))"
if errorlevel 1 (
    echo Dependencias nao instaladas. Execute: python scripts\install.py
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m aronian.web --open
if errorlevel 1 (
    echo.
    echo Nao foi possivel iniciar o jogo. Verifique a mensagem acima.
    pause
)
