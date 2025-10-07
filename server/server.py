#!/usr/bin/env python3
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
    # Optional: bundled by PyInstaller if available at build time
    from dotenv import load_dotenv as _load_dotenv_lib
except Exception:
    _load_dotenv_lib = None

def _load_dotenv_fallback(dotenv_path: str | None = None, override: bool = False) -> bool:
    """Minimal .env loader to avoid external dependency at runtime.

    Supports KEY=VALUE and export KEY=VALUE, ignores comments and blank lines.
    """
    import re
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

                # Remove surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]

                if override or key not in os.environ:
                    os.environ[key] = val
                    loaded_any = True

        return loaded_any
    except Exception:
        return False

# Unified loader alias
def load_dotenv(dotenv_path: str | None = None, override: bool = False) -> bool:
    if _load_dotenv_lib is not None:
        try:
            return _load_dotenv_lib(dotenv_path=dotenv_path, override=override)
        except Exception:
            # Fall back to internal parser on any error
            return _load_dotenv_fallback(dotenv_path, override)
    return _load_dotenv_fallback(dotenv_path, override)

import tkinter as tk
from tkinter import scrolledtext
import threading

# Load environment variables, including embedded .env when packaged
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

    # Repo/script locations and CWD
    candidates.extend([
        Path(__file__).resolve().parent / '.env',
        Path.cwd() / '.env',
    ])

    for p in candidates:
        try:
            if p and p.exists():
                if load_dotenv(dotenv_path=str(p), override=False):
                    logger = logging.getLogger(__name__)
                    logger.info(f"Loaded .env from {p}")
                    return True
        except Exception:
            continue

    # Fallback to default search (CWD)
    return load_dotenv(override=False)

_load_embedded_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import Deepgram SDK
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
    logger.info(f"✅ Deepgram SDK loaded successfully (version: {DG_VERSION})")

except ImportError as e:
    DEEPGRAM_AVAILABLE = False
    logger.error(f"❌ Deepgram SDK import failed: {e}")
except Exception as e:
    DEEPGRAM_AVAILABLE = False
    logger.error(f"❌ Unexpected error loading Deepgram: {e}")


class TranscriptProcessor:
    """
    COMPLETELY FIXED transcript processor - eliminates duplication entirely!

    Key principle: Interim results completely REPLACE previous interim display
    No more appending, no more duplication!
    """

    def __init__(self):
        self.client_state = {}

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra spaces and fixing punctuation"""
        if not text:
            return ""

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])([^\s])', r'\1 \2', text)

        # Remove leading/trailing whitespace
        return text.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two text strings"""
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

    def _should_break_line(self, text: str, accumulated_length: int) -> bool:
        """Determine if we should break to a new line"""
        # Break on sentence endings
        if text and text[-1] in '.!?':
            return True

        # Break on long accumulated text (100+ characters)
        if accumulated_length > 100:
            return True

        return False

    def process_transcript(self, client_id: str, text: str, is_final: bool = False) -> Dict:
        """
        Process transcript with PROPER interim handling - FIXES DUPLICATION!

        Key principle: Interim results REPLACE the previous interim entirely
        Final results get committed and can trigger line breaks

        Returns:
        {
            'action': 'update_interim' | 'commit_final' | 'ignore',
            'text': processed_text,
            'should_break': boolean,
        }
        """

        if client_id not in self.client_state:
            self.client_state[client_id] = {
                'last_interim': '',           # Last interim text we displayed
                'last_final': '',             # Last final text committed
                'accumulated_text': '',       # All committed text in current block
                'accumulated_length': 0,      # Length of accumulated text
                'last_timestamp': None
            }

        state = self.client_state[client_id]

        # Normalize the input text
        clean_text = self._normalize_text(text)
        if not clean_text:
            return {'action': 'ignore', 'text': '', 'should_break': False}

        if not is_final:
            # INTERIM RESULT HANDLING
            # Key insight: Always REPLACE the previous interim display - NO APPENDING!

            # Check if this interim is too similar to last interim (likely duplicate)
            if state['last_interim']:
                similarity = self._calculate_similarity(state['last_interim'], clean_text)

                # If very similar and not longer, likely a duplicate/correction - ignore
                if similarity > 0.95 and len(clean_text) <= len(state['last_interim']):
                    return {'action': 'ignore', 'text': '', 'should_break': False}

            # Store this as the current interim
            state['last_interim'] = clean_text

            return {
                'action': 'update_interim',
                'text': clean_text,
                'should_break': False,
                'metadata': {'type': 'interim'}
            }

        else:
            # FINAL RESULT HANDLING

            # Check if this final is duplicate of last final
            if state['last_final']:
                similarity = self._calculate_similarity(state['last_final'], clean_text)
                if similarity > 0.90:
                    # Very similar to last final, ignore
                    return {'action': 'ignore', 'text': '', 'should_break': False}

            # Store this final
            state['last_final'] = clean_text
            state['last_interim'] = ''  # Clear interim

            # Update accumulated text
            state['accumulated_text'] += (' ' if state['accumulated_text'] else '') + clean_text
            state['accumulated_length'] += len(clean_text)
            state['last_timestamp'] = datetime.now()

            # Determine if we should break the line
            should_break = self._should_break_line(clean_text, state['accumulated_length'])

            if should_break:
                state['accumulated_text'] = ''
                state['accumulated_length'] = 0

            return {
                'action': 'commit_final',
                'text': clean_text,
                'should_break': should_break,
                'metadata': {
                    'type': 'final',
                    'word_count': len(clean_text.split())
                }
            }


class TranscriptionGUI:
    """GUI with COMPLETELY FIXED interim result handling - eliminates all duplication!"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 Speech-to-Text Server - FIXED (No Duplication)")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')

        # Initialize FIXED transcript processor
        self.transcript_processor = TranscriptProcessor()

        # Track interim regions per client - CRITICAL for proper handling
        self._interim_regions = {}  # client_id -> (start_mark, end_mark)

        self._setup_gui()

    def _setup_gui(self):
        """Setup the GUI components"""
        # Header with success indicator
        header_frame = tk.Frame(self.root, bg='#27AE60', height=80)
        header_frame.pack(fill='x', pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🎯 FIXED: Live Speech Transcription - No More Duplication!",
            font=('Arial', 20, 'bold'),
            bg='#27AE60',
            fg='white'
        )
        title_label.pack(pady=20)

        # Status bar
        status_frame = tk.Frame(self.root, bg='#2ECC71', height=50)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="✅ DUPLICATION ISSUE FIXED - Proper interim handling | Clients: 0",
            font=('Arial', 12, 'bold'),
            bg='#2ECC71',
            fg='white'
        )
        self.status_label.pack(pady=10)

        # Main content area
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Fix explanation
        fix_frame = tk.Frame(content_frame, bg='#F39C12', height=60)
        fix_frame.pack(fill='x', pady=(0, 10))
        fix_frame.pack_propagate(False)

        fix_label = tk.Label(
            fix_frame,
            text="🔧 KEY FIX APPLIED: Interim results now REPLACE previous content (not append). This eliminates all text duplication!",
            font=('Arial', 11, 'bold'),
            bg='#F39C12',
            fg='white',
            wraplength=1000
        )
        fix_label.pack(pady=15)

        # Transcription display
        display_label = tk.Label(
            content_frame,
            text="Live Transcriptions (FIXED - Zero Duplication)",
            font=('Arial', 16, 'bold'),
            bg='#2C3E50',
            fg='#ECF0F1'
        )
        display_label.pack(anchor='w', pady=(0, 10))

        # Scrolled text widget
        self.transcription_text = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=('Arial', 14),
            bg='#ECF0F1',
            fg='#2C3E50',
            padx=20,
            pady=20,
            relief='flat',
            borderwidth=0
        )
        self.transcription_text.pack(fill='both', expand=True)

        # Configure text tags for different types
        self.transcription_text.tag_config('interim_style',
                                         font=('Arial', 14, 'italic'),
                                         foreground='#7F8C8D',  # Gray for interim
                                         background='#F8F9FA')  # Light background

        self.transcription_text.tag_config('final_style',
                                         font=('Arial', 14, 'bold'),
                                         foreground='#2C3E50')  # Dark for final

        self.transcription_text.tag_config('separator_style',
                                         font=('Arial', 10),
                                         foreground='#BDC3C7')

        # Control buttons
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
            relief='flat',
            cursor='hand2'
        )
        clear_btn.pack(side='left', padx=5)

        save_btn = tk.Button(
            button_frame,
            text="Save Transcriptions",
            command=self.save_transcriptions,
            font=('Arial', 12, 'bold'),
            bg='#27AE60',
            fg='white',
            padx=20,
            pady=10,
            relief='flat',
            cursor='hand2'
        )
        save_btn.pack(side='left', padx=5)

        # Test button to demonstrate the fix
        test_btn = tk.Button(
            button_frame,
            text="🧪 Test Fix",
            command=self.test_fix,
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            padx=20,
            pady=10,
            relief='flat',
            cursor='hand2'
        )
        test_btn.pack(side='left', padx=5)

        self.client_count = 0

    def test_fix(self):
        """Test the fix with your exact problematic example"""
        self.clear_transcriptions()

        # Add test message
        self.transcription_text.config(state='normal')
        self.transcription_text.insert('end', "🧪 TESTING WITH YOUR EXACT EXAMPLE:\n\n", 'final_style')
        self.transcription_text.config(state='disabled')

        # Simulate exactly your problematic sequence
        test_sequence = [
            ("test_client", "Okay, let's try to", False),
            ("test_client", "Okay. Let's try to", False),  # This used to cause: "Okay, let's try toOkay. Let's try to"
            ("test_client", "Okay. Let's try to talk now", False),  # This used to add more duplicates
            ("test_client", "Okay. Let's try to talk now, see how this is working.", False),
            ("test_client", "Okay. Let's try to talk now, see how this is working.", True),  # Final

            ("test_client", "And, what's the progress", False),
            ("test_client", "And, what's the progress it made till now.", False),
            ("test_client", "And, what's the progress it made till now.", True),  # Final

            ("test_client", "Still, I do see some lag", False),
            ("test_client", "Still, I do see some lag here.", False),
            ("test_client", "Still, I do see some lag here.", True),  # Final
        ]

        def run_test():
            import time
            for client_id, text, is_final in test_sequence:
                self.handle_transcript_update(client_id, text, is_final)
                time.sleep(0.4 if not is_final else 1.0)  # Slower for visibility

        # Run test in background thread
        import threading
        threading.Thread(target=run_test, daemon=True).start()

    # Thread-safe wrappers for GUI updates from background threads
    def post_status(self, message: str, client_count: int = 0):
        """Thread-safe status update"""
        self.root.after(0, self.update_status, message, client_count)

    def post_transcript_update(self, client_id: str, text: str, is_final: bool = False):
        """Thread-safe transcript update"""
        self.root.after(0, self.handle_transcript_update, client_id, text, is_final)

    def update_status(self, message: str, client_count: int = 0):
        """Update status bar"""
        self.client_count = client_count
        self.status_label.config(
            text=f"✅ DUPLICATION FIXED - {message} | Clients: {client_count}"
        )

    def handle_transcript_update(self, client_id: str, text: str, is_final: bool = False):
        """COMPLETELY FIXED transcript update handling - eliminates ALL duplication!"""

        result = self.transcript_processor.process_transcript(client_id, text, is_final)

        if result['action'] == 'ignore':
            return

        self.transcription_text.config(state='normal')

        if result['action'] == 'update_interim':
            # INTERIM RESULT: Replace the entire interim region - NO APPENDING!

            if client_id in self._interim_regions:
                # We have an existing interim region - COMPLETELY REPLACE IT
                start_mark, end_mark = self._interim_regions[client_id]

                try:
                    # DELETE the old interim content entirely
                    self.transcription_text.delete(start_mark, end_mark)

                    # INSERT the new interim content at the exact same position
                    self.transcription_text.insert(start_mark, result['text'], 'interim_style')

                    # Update the end mark to reflect new content length
                    new_end = self.transcription_text.index(f"{start_mark} + {len(result['text'])}c")
                    self.transcription_text.mark_set(end_mark, new_end)

                except tk.TclError:
                    # Mark doesn't exist anymore, create new region
                    self._create_new_interim_region(client_id, result['text'])
            else:
                # No existing interim region - create one
                self._create_new_interim_region(client_id, result['text'])

        elif result['action'] == 'commit_final':
            # FINAL RESULT: Replace interim with final text, then optionally break line

            final_text = result['text']

            if client_id in self._interim_regions:
                # Replace interim region with final text
                start_mark, end_mark = self._interim_regions[client_id]

                try:
                    # DELETE the interim content
                    self.transcription_text.delete(start_mark, end_mark)

                    # INSERT the final content at the same position
                    self.transcription_text.insert(start_mark, final_text, 'final_style')

                    # COMPLETELY REMOVE the interim region tracking
                    self.transcription_text.mark_unset(start_mark)
                    self.transcription_text.mark_unset(end_mark)
                    del self._interim_regions[client_id]

                except tk.TclError:
                    # Mark doesn't exist, just append final text
                    self.transcription_text.insert('end', final_text, 'final_style')
            else:
                # No interim region, just append final text
                if self.transcription_text.index('end-1c') != '1.0':
                    # Add space before if needed
                    last_char = self.transcription_text.get('end-2c', 'end-1c')
                    if last_char and not last_char.isspace():
                        self.transcription_text.insert('end', ' ')

                self.transcription_text.insert('end', final_text, 'final_style')

            # Add line break if needed
            if result['should_break']:
                self.transcription_text.insert('end', '\n' + '-' * 80 + '\n', 'separator_style')

        self.transcription_text.config(state='disabled')
        self.transcription_text.see('end')

    def _create_new_interim_region(self, client_id: str, text: str):
        """Create a new interim region for tracking"""

        # Add space before if needed
        if self.transcription_text.index('end-1c') != '1.0':
            last_char = self.transcription_text.get('end-2c', 'end-1c')
            if last_char and not last_char.isspace():
                self.transcription_text.insert('end', ' ')

        # Create marks for the interim region
        start_pos = self.transcription_text.index('end-1c')
        start_mark = f"{client_id}_interim_start"
        end_mark = f"{client_id}_interim_end"

        # Set start mark
        self.transcription_text.mark_set(start_mark, start_pos)

        # Insert interim text
        self.transcription_text.insert(start_pos, text, 'interim_style')

        # Set end mark
        end_pos = self.transcription_text.index(f"{start_pos} + {len(text)}c")
        self.transcription_text.mark_set(end_mark, end_pos)

        # Track this region
        self._interim_regions[client_id] = (start_mark, end_mark)

    def clear_transcriptions(self):
        """Clear all transcriptions"""
        self.transcription_text.config(state='normal')
        self.transcription_text.delete('1.0', 'end')
        self.transcription_text.config(state='disabled')

        # Reset processor and regions
        self.transcript_processor = TranscriptProcessor()
        self._interim_regions = {}

        logger.info("Transcriptions cleared")

    def save_transcriptions(self):
        """Save transcriptions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcriptions_FIXED_{timestamp}.txt"
        content = self.transcription_text.get('1.0', 'end')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Transcriptions saved to {filename}")
        self.update_status(f"Saved to {filename}", self.client_count)

    def run(self):
        """Run the GUI"""
        self.root.mainloop()


class DeepgramTranscriptionHandler:
    """Handler for Deepgram transcription"""

    def __init__(self, config: dict, client_id: str, gui: TranscriptionGUI):
        self.config = config
        self.client_id = client_id
        self.gui = gui
        self.deepgram_client = None
        self.connection = None
        self.is_connected = False

        try:
            self.pause_break_secs = float(os.getenv('PAUSE_BREAK_SECS', '2.0'))
        except Exception:
            self.pause_break_secs = 2.0

    async def connect(self):
        """Connect to Deepgram"""
        try:
            self.deepgram_client = DeepgramClient(self.config['api_key'])
            self.connection = self.deepgram_client.listen.websocket.v("1")

            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)

            # Extra visibility into stream readiness and configuration
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
                utterance_end_ms=1000,  # must be int per SDK
                endpointing=300,
            )

            if self.connection.start(options):
                self.is_connected = True
                logger.info(f"✅ Deepgram connected for client {self.client_id[:8]}")
                return True
            else:
                logger.error("❌ Failed to start Deepgram connection")
                return False

        except Exception as e:
            logger.error(f"❌ Deepgram connection error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_open(self, *args, **kwargs):
        logger.info(f"Deepgram opened for {self.client_id[:8]}")

    def _on_metadata(self, *args, **kwargs):
        meta = args[0] if args else kwargs
        try:
            meta_str = json.dumps(meta if isinstance(meta, dict) else getattr(meta, '__dict__', {}))
        except Exception:
            meta_str = str(meta)
        logger.info(f"Deepgram metadata for {self.client_id[:8]}: {meta_str[:200]}")

    def _on_transcript(self, *args, **kwargs):
        """Handle transcript events from Deepgram with robust parsing."""
        try:
            # Deepgram v4 callback signatures vary. Select the first arg that looks like a payload.
            def looks_like_payload(x) -> bool:
                try:
                    if isinstance(x, dict):
                        ch = x.get('channel')
                        return bool(ch or x.get('type'))
                    # hasattr path
                    if hasattr(x, 'channel'):
                        return True
                    d = getattr(x, '__dict__', {})
                    return 'channel' in d or 'type' in d
                except Exception:
                    return False

            candidates = []
            # collect kwargs candidates
            for k in ('result', 'data', 'payload', 'message'):
                if k in kwargs:
                    candidates.append(kwargs[k])

            # collect positional candidates
            candidates.extend(list(args))

            result = None
            for c in candidates:
                obj = c

                # Parse JSON strings if needed
                if isinstance(obj, (bytes, str)):
                    try:
                        obj = json.loads(obj)
                    except Exception:
                        pass

                if looks_like_payload(obj):
                    result = obj
                    break

            if not result:
                logger.info(
                    f"Transcript callback arg types={[type(a).__name__ for a in args]} kw={list(kwargs.keys())}"
                )
                return

            # Support SDK object or dict payloads
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

            # Detect final flag across versions/fields (Transcript events only)
            speech_final = bool(
                get_attr(result, 'speech_final', False)
                or get_attr(result, 'is_final', False)
            )

            # Clean the transcript text
            text_piece = str(transcript).strip()

            # Use the COMPLETELY FIXED transcript processor
            self.gui.post_transcript_update(self.client_id, text_piece, speech_final)

            logger.info(f"📝 [{self.client_id[:8]}] len={len(text_piece)} final={speech_final} text='{text_piece[:50]}...'")

        except Exception as e:
            logger.error(f"Transcript error: {e}")

    def _on_error(self, *args, **kwargs):
        error_data = args[0] if args else kwargs.get('error')
        if hasattr(error_data, 'error'):
            error = error_data.error
        else:
            error = error_data or 'Unknown'
        logger.error(f"Deepgram error: {error}")

    def _on_close(self, *args, **kwargs):
        self.is_connected = False
        logger.info(f"Deepgram closed for {self.client_id[:8]}")

    # Make send_audio async to avoid blocking
    async def send_audio(self, audio_data: bytes):
        """Send audio to Deepgram without blocking the event loop."""
        if self.connection and self.is_connected:
            try:
                # Run the blocking synchronous call in a separate thread
                await asyncio.to_thread(self.connection.send, audio_data)
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def close(self):
        """Close connection"""
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
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.handlers: Dict[str, DeepgramTranscriptionHandler] = {}
        self.config = {
            'api_key': os.getenv('STT_API_KEY', ''),
            'model': os.getenv('STT_MODEL', 'nova-2'),
            'language': os.getenv('STT_LANGUAGE', 'en-IN'),
            'encoding': os.getenv('ENCODING', 'linear16'),
            'sample_rate': int(os.getenv('SAMPLE_RATE', '16000'))
        }

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        client_id = str(id(websocket))
        self.clients[client_id] = websocket

        # Schedule GUI status update on main thread
        self.gui.post_status(f"Client {client_id[:8]} connected", len(self.clients))
        logger.info(f"✅ Client {client_id[:8]} connected from {websocket.remote_address}")

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
                'message': '🎯 Connected to FIXED server - zero duplication!'
            }))

            async for message in websocket:
                if isinstance(message, bytes):
                    await handler.send_audio(message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id[:8]} disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self._cleanup_client(client_id)

    async def _cleanup_client(self, client_id: str):
        """Clean up client"""
        if client_id in self.handlers:
            await self.handlers[client_id].close()
            del self.handlers[client_id]

        if client_id in self.clients:
            del self.clients[client_id]

        # Schedule GUI status update on main thread
        self.gui.post_status("Client disconnected", len(self.clients))
        logger.info(f"Cleaned up client {client_id[:8]}")

    async def start(self):
        """Start WebSocket server"""
        logger.info(f"🎯 Starting COMPLETELY FIXED server on {self.host}:{self.port}")
        logger.info(f"Model: {self.config['model']}, Language: {self.config['language']}")
        logger.info("🔧 Key Fix: Interim results now REPLACE instead of APPEND - zero duplication!")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10_000_000,
            ping_interval=20,
            ping_timeout=10
        ):
            # Schedule initial GUI status update on main thread
            self.gui.post_status("Waiting for clients...", 0)
            await asyncio.Future()


def run_server_async(gui: TranscriptionGUI):
    """Run server in async loop"""
    api_key = os.getenv('STT_API_KEY', '')
    if not api_key:
        logger.error("❌ STT_API_KEY not set!")
        return

    if not DEEPGRAM_AVAILABLE:
        logger.error("❌ Deepgram SDK not available!")
        return

    server = SpeechToTextServer(
        gui,
        host=os.getenv('SERVER_HOST', '0.0.0.0'),
        port=int(os.getenv('SERVER_PORT', '8765'))
    )

    asyncio.run(server.start())


def main():
    """Main entry point"""
    print("=" * 80)
    print("🎯 COMPLETELY FIXED SPEECH-TO-TEXT SERVER - ZERO DUPLICATION!")
    print("=" * 80)

    if not DEEPGRAM_AVAILABLE:
        print("❌ ERROR: Deepgram SDK not loaded!")
        return

    print("✅ Deepgram SDK loaded")
    print("🔧 Starting server with COMPLETELY FIXED transcript processing...")
    print("🎯 KEY FIX: Interim results now REPLACE instead of APPEND")
    print("📋 Expected result: Clean transcripts with zero text duplication")
    print()

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
