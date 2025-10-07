#!/bin/bash
# Script to build CLIENT executable for distribution

echo "===================================="
echo "Building AegisTranscribe Client"
echo "===================================="
echo

echo "Installing requirements..."
pip install -r requirements_client.txt

echo
echo "Building executable..."
pyinstaller --onefile --windowed --name "AegisTranscribeClient" \
    --add-data ".:." \
    --hidden-import=websockets \
    --hidden-import=pyaudio \
    client.py

echo
echo "===================================="
echo "Build complete!"
echo "===================================="
echo
echo "Executable location: dist/AegisTranscribeClient"
echo
echo "You can now distribute the executable to users"
echo "Users just need to run it - no Python required!"
echo
