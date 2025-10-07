@echo off
REM Build SERVER executable for Windows
setlocal enableextensions enabledelayedexpansion

echo ====================================
echo Building AegisTranscribe Server (Windows)
echo ====================================

echo 🧹 Cleaning up previous builds...
IF EXIST build ( rmdir /s /q build )
IF EXIST dist ( rmdir /s /q dist )
IF EXIST AegisTranscribeServer.spec ( del AegisTranscribeServer.spec )
echo Cleanup complete.
echo.


where pyinstaller >NUL 2>&1
if %errorlevel% neq 0 (
  echo Installing PyInstaller...
  pip install --upgrade pip >NUL
  pip install pyinstaller >NUL
)

echo Installing server requirements...
pip install -r requirements_server.txt

echo.
echo Building executable (console enabled for logs)...
pyinstaller --onefile --name "AegisTranscribeServer" ^
  --add-data .env;. ^
  --hidden-import websockets ^
  --hidden-import deepgram ^
  --hidden-import dotenv ^
  server.py

echo.
echo ====================================
echo Build complete!
echo ====================================
echo.
echo Executable: dist\AegisTranscribeServer.exe
echo Notes:
echo  - Ensure a .env with STT_API_KEY is alongside the EXE or in CWD
echo  - Optional .env: SERVER_HOST, SERVER_PORT, STT_MODEL, STT_LANGUAGE, ENCODING, SAMPLE_RATE, PAUSE_BREAK_SECS
echo.
endlocal
