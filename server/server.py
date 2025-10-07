#!/usr/bin/env python3

"""
SPEECH-TO-TEXT SERVER 
Compatible with deepgram-sdk 4.1.0+
"""

import asyncio
import websockets
import json
import logging
from typing import Dict
from datetime import datetime
import os
import sys
from pathlib import Path
import re

try:
    from dotenv import load_dotenv as _load_dotenv_lib
except Exception:
    _load_dotenv_lib = None

def _load_dotenv_fallback(dotenv_path: str | None = None, override: bool = False) -> bool:
    """Minimal .env loader"""
    path = None
    try:
        if dotenv_path:
            path = Path(dotenv_path)
        else:
            path = Path.cwd() / '.env'

        if not path.exists():
            return False

        loaded_any = False
        with path.open('r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue

                if line.lower().startswith('export '):
                    line = line[7:].strip()

                if '=' not in line:
                    continue

                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()

                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]

                if override or key not in os.environ:
                    os.environ[key] = val
                    loaded_any = True

        return loaded_any
    except Exception:
        return False

def load_dotenv(dotenv_path: str | None = None, override: bool = False) -> bool:
    if _load_dotenv_lib is not None:
        try:
            return _load_dotenv_lib(dotenv_path=dotenv_path, override=override)
        except Exception:
            return _load_dotenv_fallback(dotenv_path, override)
    return _load_dotenv_fallback(dotenv_path, override)

import tkinter as tk
from tkinter import scrolledtext
import threading

def _load_embedded_env() -> bool:
    candidates = []
    try:
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            candidates.append(Path(meipass) / '.env')
    except Exception:
        pass

    try:
        candidates.append(Path(sys.executable).parent / '.env')
    except Exception:
        pass

    candidates.extend([
        Path(__file__).resolve().parent / '.env',
        Path.cwd() / '.env',
    ])

    for p in candidates:
        try:
            if p and p.exists():
                if load_dotenv(dotenv_path=str(p), override=False):
                    return True
        except Exception:
            continue

    return load_dotenv(override=False)

_load_embedded_env()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from deepgram import (
        DeepgramClient,
        LiveTranscriptionEvents,
        LiveOptions,
    )

    try:
        from deepgram import __version__ as DG_VERSION
    except Exception:
        DG_VERSION = "unknown"

    DEEPGRAM_AVAILABLE = True
    logger.info(f"Deepgram SDK loaded successfully (version: {DG_VERSION})")

except ImportError as e:
    DEEPGRAM_AVAILABLE = False
    logger.error(f"Deepgram SDK import failed: {e}")
except Exception as e:
    DEEPGRAM_AVAILABLE = False
    logger.error(f"Unexpected error loading Deepgram: {e}")


class TranscriptProcessor:
    """Transcript processor - NO DUPLICATION"""

    def __init__(self):
        self.client_state = {}

    def _normalize_text(self, text: str) -> str:
        """Normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)
        return text.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity"""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def process_transcript(self, client_id: str, text: str, is_final: bool = False) -> Dict:
        """Process transcript"""

        if client_id not in self.client_state:
            self.client_state[client_id] = {
                'last_interim': '',
                'last_final': '',
                'accumulated_length': 0,
            }

        state = self.client_state[client_id]
        clean_text = self._normalize_text(text)

        if not clean_text:
            return {'action': 'ignore', 'text': ''}

        if not is_final:
            # INTERIM
            if state['last_interim']:
                similarity = self._calculate_similarity(state['last_interim'], clean_text)
                if similarity > 0.95 and len(clean_text) <= len(state['last_interim']):
                    return {'action': 'ignore', 'text': ''}

            state['last_interim'] = clean_text
            return {'action': 'update_interim', 'text': clean_text}
        else:
            # FINAL
            if state['last_final']:
                similarity = self._calculate_similarity(state['last_final'], clean_text)
                if similarity > 0.90:
                    return {'action': 'ignore', 'text': ''}

            state['last_final'] = clean_text
            state['last_interim'] = ''
            state['accumulated_length'] += len(clean_text)

            should_break = (clean_text.endswith('.') or clean_text.endswith('?') or 
                          clean_text.endswith('!') or state['accumulated_length'] > 100)

            if should_break:
                state['accumulated_length'] = 0

            return {'action': 'commit_final', 'text': clean_text, 'should_break': should_break}


class TranscriptionGUI:
    """GUI with PROPER deletion - no styling tricks"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Speech-to-Text Server - FIXED")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')

        self.transcript_processor = TranscriptProcessor()
        self._interim_positions = {}  # Track interim text positions

        self._setup_gui()

    def _setup_gui(self):
        """Setup GUI"""
        # Header
        header_frame = tk.Frame(self.root, bg='#27AE60', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="FIXED: Speech-to-Text Server (No Duplication)",
            font=('Arial', 20, 'bold'),
            bg='#27AE60',
            fg='white'
        )
        title_label.pack(pady=20)

        # Status
        status_frame = tk.Frame(self.root, bg='#2ECC71', height=50)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="FIXED - Interim text is properly deleted | Clients: 0",
            font=('Arial', 12, 'bold'),
            bg='#2ECC71',
            fg='white'
        )
        self.status_label.pack(pady=10)

        # Content
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Transcription display
        self.transcription_text = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=('Arial', 14),
            bg='#ECF0F1',
            fg='#2C3E50',
            padx=20,
            pady=20,
            relief='flat'
        )
        self.transcription_text.pack(fill='both', expand=True)

        # Buttons
        button_frame = tk.Frame(self.root, bg='#2C3E50')
        button_frame.pack(fill='x', padx=20, pady=10)

        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_transcriptions,
            font=('Arial', 12, 'bold'),
            bg='#E74C3C',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        clear_btn.pack(side='left', padx=5)

        save_btn = tk.Button(
            button_frame,
            text="Save",
            command=self.save_transcriptions,
            font=('Arial', 12, 'bold'),
            bg='#27AE60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        save_btn.pack(side='left', padx=5)

        self.client_count = 0

    def post_status(self, message: str, client_count: int = 0):
        self.root.after(0, self.update_status, message, client_count)

    def post_transcript_update(self, client_id: str, text: str, is_final: bool = False):
        self.root.after(0, self.handle_transcript_update, client_id, text, is_final)

    def update_status(self, message: str, client_count: int = 0):
        self.client_count = client_count
        self.status_label.config(text=f"FIXED - {message} | Clients: {client_count}")

    def handle_transcript_update(self, client_id: str, text: str, is_final: bool = False):
        """TRULY FIXED - properly deletes interim text"""

        result = self.transcript_processor.process_transcript(client_id, text, is_final)

        if result['action'] == 'ignore':
            return

        self.transcription_text.config(state='normal')

        try:
            if result['action'] == 'update_interim':
                # INTERIM: Delete old interim if exists, insert new one

                if client_id in self._interim_positions:
                    # Delete old interim text
                    start_pos, end_pos = self._interim_positions[client_id]
                    self.transcription_text.delete(start_pos, end_pos)

                    # Insert new interim at same position
                    self.transcription_text.insert(start_pos, result['text'])

                    # Update end position
                    new_end = self.transcription_text.index(f"{start_pos} + {len(result['text'])}c")
                    self._interim_positions[client_id] = (start_pos, new_end)
                else:
                    # Create new interim region
                    # Add space if needed
                    if self.transcription_text.index('end-1c') != '1.0':
                        last_char = self.transcription_text.get('end-2c', 'end-1c')
                        if last_char and not last_char.isspace():
                            self.transcription_text.insert('end', ' ')

                    start_pos = self.transcription_text.index('end-1c')
                    self.transcription_text.insert(start_pos, result['text'])
                    end_pos = self.transcription_text.index(f"{start_pos} + {len(result['text'])}c")
                    self._interim_positions[client_id] = (start_pos, end_pos)

            elif result['action'] == 'commit_final':
                # FINAL: Delete interim if exists, insert final

                final_text = result['text']

                if client_id in self._interim_positions:
                    # Delete interim text
                    start_pos, end_pos = self._interim_positions[client_id]
                    self.transcription_text.delete(start_pos, end_pos)

                    # Insert final at same position
                    self.transcription_text.insert(start_pos, final_text)

                    # Remove interim tracking
                    del self._interim_positions[client_id]
                else:
                    # No interim - just append final
                    if self.transcription_text.index('end-1c') != '1.0':
                        last_char = self.transcription_text.get('end-2c', 'end-1c')
                        if last_char and not last_char.isspace():
                            self.transcription_text.insert('end', ' ')
                    self.transcription_text.insert('end', final_text)

                # Add line break if needed
                if result.get('should_break', False):
                    self.transcription_text.insert('end', '\n' + '-' * 80 + '\n')

        except tk.TclError as e:
            logger.error(f"TclError: {e} - resetting interim tracking")
            if client_id in self._interim_positions:
                del self._interim_positions[client_id]

        self.transcription_text.config(state='disabled')
        self.transcription_text.see('end')

    def clear_transcriptions(self):
        self.transcription_text.config(state='normal')
        self.transcription_text.delete('1.0', 'end')
        self.transcription_text.config(state='disabled')
        self.transcript_processor = TranscriptProcessor()
        self._interim_positions = {}
        logger.info("Transcriptions cleared")

    def save_transcriptions(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcriptions_{timestamp}.txt"
        content = self.transcription_text.get('1.0', 'end')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved to {filename}")
        self.update_status(f"Saved to {filename}", self.client_count)

    def run(self):
        self.root.mainloop()


class DeepgramTranscriptionHandler:
    """Handler for Deepgram"""

    def __init__(self, config: dict, client_id: str, gui: TranscriptionGUI):
        self.config = config
        self.client_id = client_id
        self.gui = gui
        self.deepgram_client = None
        self.connection = None
        self.is_connected = False

    async def connect(self):
        try:
            self.deepgram_client = DeepgramClient(self.config['api_key'])
            self.connection = self.deepgram_client.listen.websocket.v("1")

            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            try:
                self.connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            except Exception:
                pass
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)

            options = LiveOptions(
                model=self.config['model'],
                language=self.config['language'],
                encoding=self.config['encoding'],
                sample_rate=self.config['sample_rate'],
                channels=1,
                punctuate=True,
                smart_format=True,
                interim_results=True,
                utterance_end_ms=1000,
                endpointing=300,
            )

            if self.connection.start(options):
                self.is_connected = True
                logger.info(f"Deepgram connected for client {self.client_id[:8]}")
                return True
            else:
                logger.error("Failed to start Deepgram")
                return False

        except Exception as e:
            logger.error(f"Deepgram connection error: {e}")
            return False

    def _on_open(self, *args, **kwargs):
        logger.info(f"Deepgram opened for {self.client_id[:8]}")

    def _on_metadata(self, *args, **kwargs):
        pass

    def _on_transcript(self, *args, **kwargs):
        try:
            def looks_like_payload(x) -> bool:
                try:
                    if isinstance(x, dict):
                        return bool(x.get('channel') or x.get('type'))
                    if hasattr(x, 'channel'):
                        return True
                    d = getattr(x, '__dict__', {})
                    return 'channel' in d or 'type' in d
                except Exception:
                    return False

            candidates = []
            for k in ('result', 'data', 'payload', 'message'):
                if k in kwargs:
                    candidates.append(kwargs[k])
            candidates.extend(list(args))

            result = None
            for c in candidates:
                obj = c
                if isinstance(obj, (bytes, str)):
                    try:
                        obj = json.loads(obj)
                    except Exception:
                        pass
                if looks_like_payload(obj):
                    result = obj
                    break

            if not result:
                return

            def get_attr(obj, name, default=None):
                return getattr(obj, name, default) if hasattr(obj, name) else (
                    obj.get(name, default) if isinstance(obj, dict) else default
                )

            channel = get_attr(result, 'channel', None)
            if channel is None and isinstance(result, dict):
                channel = result.get('channel')
            if channel is None:
                return

            alternatives = get_attr(channel, 'alternatives', []) or []
            alt0 = alternatives[0] if alternatives else None
            transcript = get_attr(alt0, 'transcript', '') if alt0 is not None else ''

            if not transcript or not str(transcript).strip():
                return

            speech_final = bool(
                get_attr(result, 'speech_final', False) or
                get_attr(result, 'is_final', False)
            )

            text_piece = str(transcript).strip()
            self.gui.post_transcript_update(self.client_id, text_piece, speech_final)

        except Exception as e:
            logger.error(f"Transcript error: {e}")

    def _on_error(self, *args, **kwargs):
        error_data = args[0] if args else kwargs.get('error')
        error = error_data.error if hasattr(error_data, 'error') else (error_data or 'Unknown')
        logger.error(f"Deepgram error: {error}")

    def _on_close(self, *args, **kwargs):
        self.is_connected = False
        logger.info(f"Deepgram closed for {self.client_id[:8]}")

    async def send_audio(self, audio_data: bytes):
        if self.connection and self.is_connected:
            try:
                await asyncio.to_thread(self.connection.send, audio_data)
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def close(self):
        if self.connection:
            try:
                self.connection.finish()
                self.is_connected = False
            except Exception as e:
                logger.error(f"Error closing: {e}")


class SpeechToTextServer:
    """WebSocket server"""

    def __init__(self, gui: TranscriptionGUI, host: str = "0.0.0.0", port: int = 8765):
        self.gui = gui
        self.host = host
        self.port = port
        self.clients = {}
        self.handlers = {}
        self.config = {
            'api_key': os.getenv('STT_API_KEY', ''),
            'model': os.getenv('STT_MODEL', 'nova-2'),
            'language': os.getenv('STT_LANGUAGE', 'en-IN'),
            'encoding': os.getenv('ENCODING', 'linear16'),
            'sample_rate': int(os.getenv('SAMPLE_RATE', '16000'))
        }

    async def handle_client(self, websocket, path):
        client_id = str(id(websocket))
        self.clients[client_id] = websocket

        self.gui.post_status(f"Client {client_id[:8]} connected", len(self.clients))
        logger.info(f"Client {client_id[:8]} connected")

        handler = DeepgramTranscriptionHandler(self.config, client_id, self.gui)

        try:
            if not await handler.connect():
                logger.error("Failed to connect to Deepgram")
                await websocket.close(code=1011, reason="Failed to connect to Deepgram")
                return

            self.handlers[client_id] = handler

            await websocket.send(json.dumps({
                'type': 'connected',
                'client_id': client_id,
                'message': 'Connected - no duplication!'
            }))

            async for message in websocket:
                if isinstance(message, bytes):
                    await handler.send_audio(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id[:8]} disconnected")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            await self._cleanup_client(client_id)

    async def _cleanup_client(self, client_id: str):
        if client_id in self.handlers:
            await self.handlers[client_id].close()
            del self.handlers[client_id]
        if client_id in self.clients:
            del self.clients[client_id]
        self.gui.post_status("Client disconnected", len(self.clients))

    async def start(self):
        logger.info(f"Starting server on {self.host}:{self.port}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10_000_000,
            ping_interval=20,
            ping_timeout=10
        ):
            self.gui.post_status("Waiting for clients...", 0)
            await asyncio.Future()


def run_server_async(gui: TranscriptionGUI):
    api_key = os.getenv('STT_API_KEY', '')
    if not api_key:
        logger.error("STT_API_KEY not set!")
        return

    if not DEEPGRAM_AVAILABLE:
        logger.error("Deepgram SDK not available!")
        return

    server = SpeechToTextServer(
        gui,
        host=os.getenv('SERVER_HOST', '0.0.0.0'),
        port=int(os.getenv('SERVER_PORT', '8765'))
    )

    asyncio.run(server.start())


def main():
    print("=" * 80)
    print("FIXED SPEECH-TO-TEXT SERVER")
    print("=" * 80)

    if not DEEPGRAM_AVAILABLE:
        print("ERROR: Deepgram SDK not loaded!")
        return

    print("Deepgram SDK loaded")
    print("Starting server...")

    gui = TranscriptionGUI()
    server_thread = threading.Thread(
        target=run_server_async,
        args=(gui,),
        daemon=True
    )

    server_thread.start()
    gui.run()


if __name__ == "__main__":
    main()
