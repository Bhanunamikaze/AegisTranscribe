# AegisTranscribe

Real-time mic-to-text with desktop GUIs. AegisTranscribe streams microphone audio from a client to a server over WebSocket and transcribes it live using Deepgram.

The project is split into two apps:

- Server: Receives audio over WebSocket, streams it to Deepgram for transcription, and renders a live transcript in a Tkinter GUI.
- Client: Captures microphone audio and streams it to the server over WebSocket via a simple Tkinter GUI.

Both apps are pure Python and can be packaged into single-file executables for easy distribution.

## Features
- Real-time streaming with interim and final captions (no duplication; interim text is properly replaced).
- GUI for both client and server (Tkinter).
- Multiple client support; running stats and basic controls (save/clear transcript).
- Cross-platform Python (Windows, macOS, Linux).
- Simple packaging scripts using PyInstaller.

## Repo Layout
```
.
├── Client/
│   ├── client.py                  # Client GUI: capture mic and stream to server
│   ├── requirements_client.txt    # Client dependencies
│   ├── build_client_exe.bat       # Build client EXE (Windows)
│   └── build_client_exe.sh        # Build client binary (Linux/macOS)
├── Server/
│   ├── server.py                  # Server GUI: websocket + Deepgram transcription
│   ├── requirements_server.txt    # Server dependencies
│   ├── build_server_exe.bat       # Build server EXE (Windows)
│   └── build_server_exe.sh        # Build server binary (Linux/macOS)
│   └── .env                       # Env variables                            
```

## Prerequisites
- Python 3.9+ recommended
- Deepgram API key for live transcription
- Microphone access on the client machine
- Port 8765 open between client and server (configurable)
- For PyAudio:
  - Windows: `pip install pyaudio` usually works; if not, use a prebuilt wheel
  - Linux: `sudo apt-get install -y portaudio19-dev` then `pip install pyaudio`
  - macOS: `brew install portaudio` then `pip install pyaudio`

## Deepgram (Speech Provider)
- Uses Deepgram (https://deepgram.com/) for real-time transcription.
- Particularly strong for Indian English; offers $200 free usage on signup without a credit card.
- Supports multiple models: nova-3, nova-2, nova, whisper.
- Supports 30+ languages; you can change both model and language via the `.env` file.
- Recommended for Indian English: `STT_MODEL=nova-2` or `STT_MODEL=nova-3`, with `STT_LANGUAGE=en-IN`.

## Environment Variables
Create a `.env` alongside the app you will run (root or inside Server/ when packaging). Do not commit real keys.

```
# Deepgram config
STT_API_KEY=YOUR_DEEPGRAM_API_KEY
STT_MODEL=nova-2
STT_LANGUAGE=en-IN

# Server settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8765

# Audio settings
SAMPLE_RATE=16000
ENCODING=linear16
```

## Quick Start

### 1) Run the Server
- Install dependencies: `pip install -r Server/requirements_server.txt`
- Ensure `.env` has a valid `STT_API_KEY`.
- Start: `python Server/server.py`

The server opens a GUI and listens for WebSocket clients (default `ws://0.0.0.0:8765`).

### 2) Run the Client
- Install dependencies: `pip install -r Client/requirements_client.txt`
- Start: `python Client/client.py`
- In the client GUI, enter the server address, e.g. `ws://<server-ip>:8765`, click Connect, then Start Recording.

## Networking
- Same network: The client and server must be able to reach each other on the network (same LAN/Wi‑Fi/VLAN). Use the server machine’s IP and open port `8765`.
- Over the internet (recommended approach): Use Tailscale VPN. Install Tailscale on both machines, sign in to the same tailnet, then use the server’s Tailscale IP (in the `100.x.x.x` range) in the client, e.g. `ws://100.x.x.x:8765`. This allows secure connectivity without manual port forwarding.
- Alternatives: Expose the server publicly with port forwarding and TLS (`wss://`) plus authentication and firewall rules. This requires more setup and is not recommended for quick usage.

## Build Binaries (Optional)
You can package each app into a single-file executable using PyInstaller.

- Server (Windows): `Server/build_server_exe.bat`
- Server (Linux/macOS): `bash Server/build_server_exe.sh`
- Client (Windows): `Client/build_client_exe.bat`
- Client (Linux/macOS): `bash Client/build_client_exe.sh`

Outputs are written to the `dist/` folder. Keep a `.env` with your variables next to the server binary or run it from a directory containing `.env`.

## Usage Notes
- Client captures mic at 16 kHz mono PCM and streams raw chunks to the server.
- Server relays audio to Deepgram via the official SDK, handles interim/final messages, and updates the GUI transcript without duplicating text.
- The server GUI can save the transcript to a timestamped `.txt` file and clear the view.

## Troubleshooting
- Cannot connect: verify the server is running, the IP/port is correct, and firewall allows inbound connections on the configured port (default 8765).
- No microphone: ensure PyAudio is installed and the mic is selected in the client GUI. On Linux/macOS install system PortAudio as noted above.
- Deepgram errors: confirm `STT_API_KEY` and network connectivity from the server host.
- Packaging issues: ensure PyInstaller is installed and run the provided build scripts from the corresponding `Client/` or `Server/` folder.


## License
Add your chosen license for this repository (e.g., MIT). If not specified, clarify usage terms before distribution.
