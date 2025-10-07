#!/usr/bin/env python3
"""
SPEECH-TO-TEXT SERVER (Main Host)
Receives audio from clients, processes via Deepgram, displays formatted transcription
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


class TranscriptionGUI:
    """GUI for displaying transcriptions from multiple clients"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Speech-to-Text Server - Live Transcriptions")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')

        # Header
        header_frame = tk.Frame(self.root, bg='#34495E', height=80)
        header_frame.pack(fill='x', pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Live Speech Transcription Server",
            font=('Arial', 24, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        title_label.pack(pady=20)

        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495E', height=50)
        status_frame.pack(fill='x')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Server Running | Clients: 0 | Waiting for connections...",
            font=('Arial', 12),
            bg='#34495E',
            fg='#2ECC71'
        )
        self.status_label.pack(pady=10)

        # Main content area
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Transcription display
        display_label = tk.Label(
            content_frame,
            text="Live Transcriptions",
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

        # Configure text tags
        self.transcription_text.tag_config('client_header', 
                                          font=('Arial', 12, 'bold'),
                                          foreground='#3498DB',
                                          spacing1=10,
                                          spacing3=5)
        self.transcription_text.tag_config('timestamp', 
                                          font=('Arial', 10),
                                          foreground='#7F8C8D')
        self.transcription_text.tag_config('transcription', 
                                          font=('Arial', 14),
                                          foreground='#2C3E50',
                                          spacing1=5,
                                          spacing3=15)
        self.transcription_text.tag_config('separator',
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

        self.client_count = 0
        # Track interim regions per client
        self._interim_marks = {}
        # Track aggregation per client for batching finals
        self._block_state = {}

    # Thread-safe wrappers for GUI updates from background threads
    def post_status(self, message: str, client_count: int = 0):
        self.root.after(0, self.update_status, message, client_count)

    def post_transcription(self, client_id: str, text: str, is_final: bool = False):
        self.root.after(0, self.add_transcription, client_id, text, is_final)

    def post_append_interim(self, client_id: str, text: str):
        self.root.after(0, self.append_interim, client_id, text)

    def post_break(self, client_id: str):
        self.root.after(0, self.break_block, client_id)

    def update_status(self, message: str, client_count: int = 0):
        """Update status bar"""
        self.client_count = client_count
        self.status_label.config(
            text=f"🟢 Server Running | Clients: {client_count} | {message}"
        )

    def add_transcription(self, client_id: str, text: str, is_final: bool = False):
        """Add or update transcription in display.
        - Interim (is_final=False): replace/update a single line per client.
        - Final (is_final=True): append finalized block and clear interim region.
        """
        self.transcription_text.config(state='normal')

        start_mark = f"{client_id}_interim_start"
        end_mark = f"{client_id}_interim_end"

        if not is_final:
            # Update/replace interim text without adding newlines
            if start_mark not in self.transcription_text.mark_names():
                insert_at = self.transcription_text.index('end-1c')
                self.transcription_text.mark_set(start_mark, insert_at)
                self.transcription_text.insert(insert_at, text, 'transcription')
                self.transcription_text.mark_set(end_mark, self.transcription_text.index('end-1c'))
            else:
                # Replace the existing interim region with exactly the provided text
                self.transcription_text.delete(start_mark, end_mark)
                self.transcription_text.insert(start_mark, text, 'transcription')
                self.transcription_text.mark_set(end_mark, self.transcription_text.index('end-1c'))

        else:
            # Finalize: append to ongoing block; avoid re-inserting interim content
            using_interim_region = False
            final_text = text
            if not final_text and start_mark in self.transcription_text.mark_names():
                # We already drew this interim text; use it for counters and optional move
                try:
                    final_text = self.transcription_text.get(start_mark, end_mark)
                    using_interim_region = True
                except Exception:
                    final_text = ''

            final_text = (final_text or '').strip()
            if final_text:
                now = datetime.now()
                state = self._block_state.get(client_id, {'chars': 0, 'sentences': 0, 'last_ts': None})

                # Normal path: insert text only if not already on screen
                if not using_interim_region:
                    starting_new_block = state['chars'] == 0 and state['sentences'] == 0
                    if starting_new_block:
                        if self.transcription_text.index('end-1c') != '1.0':
                            self.transcription_text.insert('end-1c', "\n")
                    else:
                        if self.transcription_text.index('end-1c') != '1.0':
                            prev_char = self.transcription_text.get('end-2c', 'end-1c')
                            if prev_char and not prev_char.isspace():
                                self.transcription_text.insert('end-1c', ' ')
                    self.transcription_text.insert('end-1c', final_text, 'transcription')

                # Ensure a trailing space to continue block if no explicit break
                if self.transcription_text.index('end-1c') != '1.0':
                    prev_char = self.transcription_text.get('end-2c', 'end-1c')
                    if prev_char and not prev_char.isspace():
                        self.transcription_text.insert('end-1c', ' ')

                # Update block state and check thresholds
                state['chars'] += len(final_text)
                state['sentences'] += sum(final_text.count(c) for c in '.!?')
                state['last_ts'] = now

                MIN_SENTENCES = 4
                if state['sentences'] >= MIN_SENTENCES:
                    logger.info(f"Threshold break: chars={state['chars']} sentences={state['sentences']}")
                    self.transcription_text.insert('end-1c', "\n" + "-" * 80 + "\n", 'separator')
                    state = {'chars': 0, 'sentences': 0, 'last_ts': now}

                self._block_state[client_id] = state

            # Finally, clear any remaining marks for this client
            for m in (start_mark, end_mark):
                try:
                    self.transcription_text.mark_unset(m)
                except Exception:
                    pass

        self.transcription_text.config(state='disabled')
        self.transcription_text.see('end')

    def break_block(self, client_id: str):
        """Force a new block for a client (pause detected)."""
        self.transcription_text.config(state='normal')
        if self.transcription_text.index('end-1c') != '1.0':
            self.transcription_text.insert('end-1c', "\n" + "-" * 80 + "\n", 'separator')
        # Reset counters for the client
        self._block_state[client_id] = {'chars': 0, 'sentences': 0, 'last_ts': None}
        self.transcription_text.config(state='disabled')
        self.transcription_text.see('end')

    def append_interim(self, client_id: str, text: str):
        """Append text to the client's interim region (no newlines)."""
        if not text:
            return
        self.transcription_text.config(state='normal')
        start_mark = f"{client_id}_interim_start"
        end_mark = f"{client_id}_interim_end"
        if start_mark not in self.transcription_text.mark_names():
            insert_at = self.transcription_text.index('end-1c')
            self.transcription_text.mark_set(start_mark, insert_at)
            self.transcription_text.insert(insert_at, text, 'transcription')
            self.transcription_text.mark_set(end_mark, self.transcription_text.index('end-1c'))
        else:
            self.transcription_text.insert(end_mark, text, 'transcription')
            self.transcription_text.mark_set(end_mark, self.transcription_text.index('end-1c'))
        self.transcription_text.config(state='disabled')
        self.transcription_text.see('end')

    def _format_transcription(self, text: str) -> str:
        """Format transcription"""
        sentences = text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')
        words = sentences.split()
        if len(words) > 50:
            formatted = []
            current = []
            for i, word in enumerate(words):
                current.append(word)
                if (i + 1) % 50 == 0:
                    formatted.append(' '.join(current))
                    formatted.append('\n')
                    current = []
            if current:
                formatted.append(' '.join(current))
            return '\n'.join(formatted)
        return sentences

    def clear_transcriptions(self):
        """Clear all transcriptions"""
        self.transcription_text.config(state='normal')
        self.transcription_text.delete('1.0', 'end')
        self.transcription_text.config(state='disabled')
        logger.info("Transcriptions cleared")

    def save_transcriptions(self):
        """Save transcriptions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcriptions_{timestamp}.txt"

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
        self.current_text = ""
        self.prev_interim = ""
        self.last_final_norm = ""
        self.last_final_ts = None
        try:
            self.pause_break_secs = float(os.getenv('PAUSE_BREAK_SECS', '2.0'))
        except Exception:
            self.pause_break_secs = 2.0

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        # Collapse multiple spaces and fix space before punctuation
        import re
        t = re.sub(r"\s+", " ", text)
        # Remove space before punctuation and close brackets
        t = re.sub(r"\s+([,.:;!?])", r"\1", t)
        t = re.sub(r"\s+([)\]}])", r"\1", t)
        # Ensure a single space after punctuation when followed by non-space
        t = re.sub(r"([,.:;!?])([^\s])", r"\1 \2", t)
        return t.strip()

    @classmethod
    def _dedupe_text(cls, text: str) -> str:
        """Remove immediate duplicated words and consecutive repeated n-grams.
        Conservative but effective for ASR interim expansions.
        """
        import re
        if not text:
            return text

        # Normalize spaces first
        text = cls._normalize_spaces(text)

        # Tokenize preserving punctuation as part of tokens
        tokens = text.split()
        if not tokens:
            return text

        def canon(tok: str) -> str:
            return tok.lower().rstrip(".,!?;:")

        # 1) Remove immediate duplicate words (case-insensitive, ignore trailing punctuation)
        dedup_words = []
        last_c = None
        for tok in tokens:
            c = canon(tok)
            if c == last_c:
                continue
            dedup_words.append(tok)
            last_c = c

        # 2) Remove consecutive repeated n-grams (n=4..2), using canonical comparison and sliding window
        def dedupe_ngrams(seq, n):
            out = []
            i = 0
            L = len(seq)
            while i < L:
                # If enough tokens for 2 n-grams, compare windows
                if i + 2*n <= L:
                    win1 = seq[i:i+n]
                    win2 = seq[i+n:i+2*n]
                    if [canon(t) for t in win1] == [canon(t) for t in win2]:
                        # emit once and skip repeats
                        out.extend(win1)
                        i += n
                        while i + n <= L and [canon(t) for t in seq[i:i+n]] == [canon(t) for t in win1]:
                            i += n
                        continue
                # No repeat; emit single token and advance
                out.append(seq[i])
                i += 1
            return out

        for n in (4, 3, 2):
            dedup_words = dedupe_ngrams(dedup_words, n)

        cleaned = " ".join(dedup_words)

        # Final pass: collapse residual duplicate words introduced by join
        cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = cls._normalize_spaces(cleaned)
        return cleaned

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
                    f"Transcript callback arg types={ [type(a).__name__ for a in args] } kw={list(kwargs.keys())}"
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
                # Log unexpected payload shape for debugging
                try:
                    if isinstance(result, dict):
                        logger.info(f"Transcript payload unexpected shape: keys={list(result.keys())}")
                    else:
                        preview = getattr(result, '__dict__', {})
                        logger.info(f"Transcript payload unexpected shape: keys={list(preview.keys())}")
                except Exception:
                    pass
                return

            alternatives = get_attr(channel, 'alternatives', []) or []
            alt0 = alternatives[0] if alternatives else None
            transcript = get_attr(alt0, 'transcript', '') if alt0 is not None else ''
            if not transcript or not str(transcript).strip():
                return

            # Detect final flag across versions/fields (Transcript events only)
            # Do NOT infer final from generic event type to avoid duplicate commits
            speech_final = bool(
                get_attr(result, 'speech_final', False)
                or get_attr(result, 'is_final', False)
            )

            # Clean and dedupe the transcript text before diffing
            text_piece = self._dedupe_text(str(transcript).strip())

            if not speech_final:
                new_text = text_piece
                # Longest common prefix with previous interim
                lcp_len = 0
                for a, b in zip(self.prev_interim, new_text):
                    if a == b:
                        lcp_len += 1
                    else:
                        break

                if len(new_text) < len(self.prev_interim):
                    # Model rewrote hypothesis; replace interim line with new text
                    self.current_text = new_text
                    self.gui.post_transcription(self.client_id, self.current_text, is_final=False)
                else:
                    # Append only the delta suffix
                    suffix = new_text[lcp_len:]
                    if suffix:
                        # Ensure a space between previous content and suffix when needed
                        if self.current_text:
                            prev_char = self.current_text[-1]
                        else:
                            prev_char = ' '
                        needs_space = (not prev_char.isspace()) and (not suffix.startswith(tuple('.,!?;:)]}"\'')))  # no space before punctuation
                        if needs_space:
                            suffix = ' ' + suffix
                            self.current_text += ' '
                        if suffix:
                            self.current_text += suffix
                            self.gui.post_append_interim(self.client_id, suffix)

                self.prev_interim = new_text
                logger.info(f"Transcript recv [{self.client_id[:8]}] len={len(text_piece)} final=False")
                return

            # Final: commit the last complete transcript once
            from datetime import datetime as _dt
            now_ts = _dt.now()

            # Pause-aware break: if long gap since previous final, force break before this one
            if self.last_final_ts is not None:
                delta = (now_ts - self.last_final_ts).total_seconds()
                if delta >= self.pause_break_secs:
                    logger.info(f"Pause >= {self.pause_break_secs}s detected: {delta:.2f}s; breaking block")
                    self.gui.post_break(self.client_id)
            self.last_final_ts = now_ts

            # Prefer the SDK's final transcript (text_piece) over prior interim
            final_text = text_piece.strip()

            # If the final adds new suffix beyond the last interim, append that delta first
            try:
                existing = self.prev_interim or ""
                lcp_len = 0
                for a, b in zip(existing, final_text):
                    if a == b:
                        lcp_len += 1
                    else:
                        break
                suffix = final_text[lcp_len:]
                if suffix:
                    # ensure spacing before non-punctuating suffix
                    prev_char = self.current_text[-1] if self.current_text else ' '
                    needs_space = (not prev_char.isspace()) and (not suffix.startswith(tuple('.,!?;:)]}"\'')))
                    if needs_space:
                        suffix_to_add = ' ' + suffix
                        self.current_text += ' '
                    else:
                        suffix_to_add = suffix
                    self.current_text += suffix
                    self.gui.post_append_interim(self.client_id, suffix_to_add)
            except Exception:
                pass

            # Update trackers to the final text
            self.prev_interim = final_text
            self.current_text = final_text

            # Debounce duplicate finals: compare normalized text of the final
            norm = final_text.lower().strip()
            if norm and norm != self.last_final_norm:
                # Do not re-print final text; keep the interim line and only add separator/space
                self.gui.post_transcription(self.client_id, "", is_final=True)
                self.last_final_norm = norm
                logger.info(f"Final (separator only) [{self.client_id[:8]}] {final_text}")
            else:
                logger.info(f"Final duplicate ignored [{self.client_id[:8]}]")
            self.current_text = ""
            self.prev_interim = ""

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

    # FIX: Make send_audio async to avoid blocking
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

    async def _send_keepalives(self, handler: DeepgramTranscriptionHandler):
        """Send keepalive messages to Deepgram in the background."""
        keepalive_message = json.dumps({"type": "KeepAlive"}).encode('utf-8')
        try:
            while handler.is_connected:
                await asyncio.to_thread(handler.connection.send, keepalive_message)
                logger.info(f"Sent keepalive for client {handler.client_id[:8]}")
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass 
        except Exception as e:
            logger.error(f"Keepalive task error: {e}")

    async def handle_client(self, websocket, path):
        """Handle client connection"""
        client_id = str(id(websocket))
        self.clients[client_id] = websocket

        # Schedule GUI status update on main thread
        self.gui.post_status(f"Client {client_id[:8]} connected", len(self.clients))
        logger.info(f"✅ Client {client_id[:8]} connected from {websocket.remote_address}")

        handler = DeepgramTranscriptionHandler(self.config, client_id, self.gui)
        keepalive_task = None

        try:
            if not await handler.connect():
                logger.error("Failed to connect to Deepgram")
                await websocket.close(code=1011, reason="Failed to connect to Deepgram")
                return

            self.handlers[client_id] = handler
            # Disable custom Deepgram keepalives to avoid mixing non-audio frames
            # keepalive_task = asyncio.create_task(self._send_keepalives(handler))

            await websocket.send(json.dumps({
                'type': 'connected',
                'client_id': client_id,
                'message': 'Connected to server. Start speaking!'
            }))

            async for message in websocket:
                if isinstance(message, bytes):
                    # FIX: Await the non-blocking send_audio method
                    await handler.send_audio(message)
                    # Debug: confirm audio flow from client to Deepgram
                    #logger.info(f"Audio chunk -> Deepgram bytes={len(message)} client={client_id[:8]}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id[:8]} disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            if keepalive_task:
                keepalive_task.cancel()
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
        logger.info(f"Starting server on {self.host}:{self.port}")
        logger.info(f"Model: {self.config['model']}, Language: {self.config['language']}")

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
    print("=" * 60)
    print("SPEECH-TO-TEXT SERVER")
    print("=" * 60)

    if not DEEPGRAM_AVAILABLE:
        print("❌ ERROR: Deepgram SDK not loaded!")
        return

    print("Deepgram SDK loaded")
    print("Starting server with GUI...")

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
