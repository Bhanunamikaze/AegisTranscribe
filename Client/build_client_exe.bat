@echo off
REM Script to build CLIENT.EXE for distribution
REM Run this on Windows to create executable

echo ====================================
echo Building AegisTranscribe Client EXE
echo ====================================
echo.

echo 🧹 Cleaning up previous builds...
IF EXIST build ( rmdir /s /q build )
IF EXIST dist ( rmdir /s /q dist )
IF EXIST AegisTranscribeClient.spec ( del AegisTranscribeClient.spec )
echo Cleanup complete.
echo.

echo 🐍 Installing requirements...
pip install -r requirements_client.txt
echo.

echo 📦 Building executable...
pyinstaller --onefile --name "AegisTranscribeClient" --icon=NONE ^
    --hidden-import=websockets ^
    --hidden-import=pyaudio ^
    client.py

echo.
echo ====================================
echo ✅ Build complete!
echo ====================================
echo.
echo Executable location: dist\AegisTranscribeClient.exe
echo.
echo You can now distribute AegisTranscribeClient.exe to users.
echo.
pause
