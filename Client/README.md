# AegisTranscribe — Client

GUI application that captures microphone audio and streams it over WebSocket to the AegisTranscribe server.

## Prerequisites
- Python 3.9+
- Microphone access and drivers
- Optional (for building): PyInstaller (installed by the build scripts)
- PyAudio OS deps if needed:
  - Windows: try `pip install pyaudio` (or use a prebuilt wheel)
  - Linux: `sudo apt-get install -y portaudio19-dev` then `pip install pyaudio`
  - macOS: `brew install portaudio` then `pip install pyaudio`

## Run From Source (Development)
```bash
pip install -r requirements_client.txt
python client.py
```

## Build Executable/Binary

### Windows
```bat
build_client_exe.bat
```
- Output: `dist/AegisTranscribeClient.exe`

### Linux/macOS
```bash
chmod +x build_client_exe.sh
./build_client_exe.sh
```
- Output: `dist/AegisTranscribeClient`

## Using the Client
1. Launch the app (from source or the packaged binary).
2. Enter the server address, for example: `ws://<server-ip>:8765`.
3. Click Connect.
4. Choose a microphone from the dropdown (click Refresh if none listed).
5. Click Start Recording to stream audio to the server.
6. Use Stop Recording to end the stream; Disconnect to close the session.

Notes
- The client does not need any API keys; the server handles transcription.
- Default audio: mono, 16 kHz PCM (`paInt16`).

## Troubleshooting
- Cannot connect: verify the server is running and reachable (port 8765 by default), and firewall rules allow it.
- No microphones listed: ensure PyAudio is installed and OS permissions allow microphone access.
- Build errors: ensure PyInstaller is installed and run the build scripts from the `Client/` directory.
