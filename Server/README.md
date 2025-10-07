# AegisTranscribe — Server

WebSocket server that receives audio from clients, streams it to Deepgram for live transcription, and renders the transcript in a Tkinter GUI.

## Prerequisites
- Python 3.9+
- Internet connection (Deepgram API)
- Deepgram API key in `.env`
- For PyAudio on some systems:
  - Linux: `sudo apt-get install -y portaudio19-dev` then `pip install pyaudio`
  - macOS: `brew install portaudio` then `pip install pyaudio`

## Configuration (.env)
Create a `.env` alongside the server (root or next to the packaged binary):
```bash
# Deepgram
STT_API_KEY=YOUR_DEEPGRAM_API_KEY
STT_MODEL=nova-2
STT_LANGUAGE=en-IN

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8765

# Audio
SAMPLE_RATE=16000
ENCODING=linear16
```

## Run From Source (Development)
```bash
pip install -r requirements_server.txt
python server.py
```
The server opens a GUI and listens on `ws://SERVER_HOST:SERVER_PORT` (default `0.0.0.0:8765`).

## Build Executable/Binary

### Windows
```bat
build_server_exe.bat
```
- Output: `dist/AegisTranscribeServer.exe`

### Linux/macOS
```bash
chmod +x build_server_exe.sh
./build_server_exe.sh
```
- Output: `dist/AegisTranscribeServer`

Place a `.env` with your settings next to the binary, or run the binary from a directory that contains `.env`.

## Using the Server
1. Start the server (from source or packaged binary).
2. The GUI shows status and an empty transcript area.
3. Clients connect to `ws://<server-ip>:<port>` and stream audio.
4. The transcript updates in real time (interim text is replaced; final text is committed).
5. Use Save to write the transcript to a timestamped `.txt`; Clear to reset the view.

## Troubleshooting
- Deepgram not available: ensure `deepgram-sdk` is installed and `STT_API_KEY` is valid.
- Port in use: change `SERVER_PORT` in `.env` and restart.
- Packaging issues: ensure PyInstaller is installed and run the provided script from the `Server/` directory.

## Security Notes
- Do not commit real API keys. Keep `.env` out of version control or use placeholders.
- For internet exposure, prefer `wss://`, add authentication, and restrict inbound traffic.
