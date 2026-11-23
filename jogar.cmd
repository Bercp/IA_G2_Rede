@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente Python nao encontrado. Consulte docs\JOGAR_WEB.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m aronian.web --open
if errorlevel 1 (
    echo.
    echo Nao foi possivel iniciar o jogo. Verifique a mensagem acima.
    pause
)
