@echo off
title StockPick
cd /d "%~dp0"

set PYTHON=C:\Users\post\AppData\Local\Python\pythoncore-3.14-64\python.exe
set STREAMLIT=C:\Users\post\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe

echo.
echo ============================================
echo   StockPick - Lokal kjoring
echo ============================================
echo.

:: Sjekk om CSV allerede finnes og er ny nok (siste 7 dager)
set CSV=top_candidates_latest.csv
set RUN_SCREENER=1

if exist "%CSV%" (
    echo CSV-fil funnet. Vil du kjoere screener paa nytt?
    echo   [J] Ja, kjoer screener  (tar 20-40 min)
    echo   [N] Nei, bruk eksisterende data
    echo.
    set /p SVAR="Valg (J/N): "
    if /i "%SVAR%"=="N" set RUN_SCREENER=0
) else (
    echo Ingen CSV funnet - screener maa kjoeres foerst.
)

echo.

if "%RUN_SCREENER%"=="1" (
    echo Starter screener... Dette tar 20-40 minutter.
    echo Du kan folge med paa fremgang i dette vinduet.
    echo.
    "%PYTHON%" screener_motor.py
    if errorlevel 1 (
        echo.
        echo FEIL: Screener krasjet. Sjekk feilmeldingen over.
        pause
        exit /b 1
    )
    echo.
    echo Screener ferdig!
)

echo Aapner StockPick i nettleseren paa http://localhost:3000 ...
echo (Trykk Ctrl+C i dette vinduet for aa stoppe appen)
echo.

"%STREAMLIT%" run streamlit_app.py --server.port 3000 --server.headless false
