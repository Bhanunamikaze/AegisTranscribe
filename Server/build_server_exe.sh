#!/bin/bash
# Script to build SERVER executable/binary for distribution (Linux/macOS)
set -euo pipefail

echo "===================================="
echo "Building AegisTranscribe Server"
echo "===================================="
echo

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "Installing PyInstaller...";
  pip install --upgrade pip >/dev/null
  pip install pyinstaller >/dev/null
fi

echo "Installing server requirements..."
pip install -r requirements_server.txt

echo
echo "Building executable (console enabled for logs)..."
pyinstaller --onefile --name "AegisTranscribeServer" \
  --add-data ".env:." \
  --hidden-import=websockets \
  --hidden-import=deepgram \
  --hidden-import=dotenv \
  server.py

echo
echo "===================================="
echo "Build complete!"
echo "===================================="
echo
echo "Binary location: dist/AegisTranscribeServer"
echo "Notes:"
echo "- Ensure a .env with STT_API_KEY is alongside the binary or in CWD"
echo "- Optional .env keys: SERVER_HOST, SERVER_PORT, STT_MODEL, STT_LANGUAGE, ENCODING, SAMPLE_RATE, PAUSE_BREAK_SECS"
echo
