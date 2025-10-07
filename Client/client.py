#!/usr/bin/env python3
"""
AegisTranscribe Client
Captures microphone audio and streams to server
"""

import asyncio
import websockets
import json
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import threading
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import pyaudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - microphone won't work")

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHANNELS = 1
RATE = 16000


class AudioStreamer:
    """Captures and streams microphone audio"""

    def __init__(self):
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.available_devices = []

        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                self._detect_devices()
            except Exception as e:
                logger.error(f"Failed to initialize PyAudio: {e}")
                self.audio = None

    def _detect_devices(self):
        """Detect available audio input devices, filter out non-user devices, and remove duplicates."""
        self.available_devices.clear()
        if not self.audio:
            return

        blocklist = [
            'Microsoft Sound Mapper',
            'Primary Sound Capture Driver',
            'Stereo Mix',
            'What U Hear',
            'Loopback'
        ]

        seen_names = set()
        try:
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    device_name = device_info['name']
                    
                    # FIX: Improve de-duplication to handle truncated names
                    # By truncating the name for the check, we can catch variations.
                    short_name = device_name[:30] 
                    is_blocked = any(substring in device_name for substring in blocklist)

                    if device_info['maxInputChannels'] > 0 and short_name not in seen_names and not is_blocked:
                        self.available_devices.append({
                            'index': i,
                            'name': device_name,
                            'channels': device_info['maxInputChannels']
                        })
                        seen_names.add(short_name)
                        logger.info(f"Found unique input device: {device_name}")
                except Exception as e:
                    continue

            if not self.available_devices:
                logger.warning("No user-facing audio input devices detected after filtering")
        except Exception as e:
            logger.error(f"Error detecting devices: {e}")

    def rescan_devices(self):
        """Rescans for available audio devices."""
        logger.info("Rescanning for audio devices...")
        self._detect_devices()

    def has_devices(self):
        """Check if any audio devices are available"""
        return len(self.available_devices) > 0

    def start_recording(self, device_index=None):
        """Start capturing audio"""
        if not self.audio:
            logger.error("PyAudio not initialized")
            return False

        if not self.has_devices():
            logger.error("No audio input devices available")
            return False

        try:
            if device_index is None and self.available_devices:
                device_index = self.available_devices[0]['index']

            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
            self.is_recording = True
            logger.info(f"✅ Microphone started (device index: {device_index})")
            return True
        except IOError as e:
            logger.error(f"Failed to start microphone (IOError): {e}")
            messagebox.showerror(
                "Microphone Error",
                f"Could not open microphone. It might be in use by another application or disconnected.\n\nError: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start microphone (Generic): {e}")
            return False

    def read_audio(self) -> bytes:
        """Read audio chunk"""
        if self.stream and self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                return data
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                self.stop_recording()
                return b''
        return b''

    def stop_recording(self):
        """Stop capturing audio"""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        self.stream = None
        logger.info("Microphone stopped")

    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass


class ClientGUI:
    """GUI for the client application"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AegisTranscribe Client")
        self.root.geometry("500x750")
        self.root.configure(bg='#2C3E50')
        # Allow window resizing and add a size grip
        self.root.resizable(True, True)
        try:
            ttk.Sizegrip(self.root).place(relx=1.0, rely=1.0, anchor='se')
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # State variables
        self.is_connected = False
        self.is_recording = False
        self.websocket = None
        self.bytes_sent = 0
        self.recording_start_time = None
        
        self.event_loop = None
        self.connection_thread = None

        # Header
        header_frame = tk.Frame(self.root, bg='#34495E', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="AegisTranscribe Client",
            font=('Arial', 20, 'bold'),
            bg='#34495E',
            fg='#ECF0F1'
        )
        title_label.pack(pady=25)

        # Scrollable content container (everything below header)
        container = tk.Frame(self.root, bg='#2C3E50')
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, bg='#2C3E50', highlightthickness=0)
        v_scroll = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        content_wrapper = tk.Frame(canvas, bg='#2C3E50')
        content_window = canvas.create_window((0, 0), window=content_wrapper, anchor='nw')

        def _on_content_resize(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(content_window, width=canvas.winfo_width())

        content_wrapper.bind('<Configure>', _on_content_resize)
        canvas.bind('<Configure>', _on_content_resize)

        # Connection section
        conn_frame = tk.Frame(content_wrapper, bg='#2C3E50')
        conn_frame.pack(fill='x', padx=20, pady=20)

        tk.Label(
            conn_frame,
            text="Server Address:",
            font=('Arial', 12),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(anchor='w', pady=(0, 5))

        self.server_entry = tk.Entry(
            conn_frame,
            font=('Arial', 12),
            bg='#ECF0F1',
            fg='#2C3E50',
            relief='flat',
            bd=0
        )
        self.server_entry.insert(0, "ws://localhost:8765")
        self.server_entry.pack(fill='x', ipady=10, pady=(0, 15))

        self.connect_btn = tk.Button(
            conn_frame,
            text="Connect to Server",
            command=self.toggle_connection,
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            padx=20,
            pady=15,
            relief='flat',
            cursor='hand2',
            activebackground='#2980B9'
        )
        self.connect_btn.pack(fill='x')

        # Status section
        status_frame = tk.Frame(content_wrapper, bg='#2C3E50')
        status_frame.pack(fill='x', padx=20, pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Disconnected",
            font=('Arial', 14, 'bold'),
            bg='#2C3E50',
            fg='#E74C3C'
        )
        self.status_label.pack()

        # Audio device
        self.audio_streamer = AudioStreamer()
        device_frame = tk.Frame(content_wrapper, bg='#2C3E50')
        device_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(
            device_frame,
            text="Select Microphone:",
            font=('Arial', 12),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(anchor='w', pady=(0, 5))
        
        mic_selection_frame = tk.Frame(device_frame, bg='#2C3E50')
        mic_selection_frame.pack(fill='x')

        self.selected_mic = tk.StringVar(self.root)
        
        style = ttk.Style()
        style.configure('TMenubutton', background='#34495E', foreground='white', font=('Arial', 11), padding=5)

        self.mic_dropdown = ttk.OptionMenu(
            mic_selection_frame,
            self.selected_mic,
            "No microphones found"
        )
        self.mic_dropdown.pack(side='left', fill='x', expand=True, ipady=5)
        
        self.refresh_mics_btn = tk.Button(
            mic_selection_frame,
            text="Refresh",
            command=self.refresh_mic_list,
            font=('Arial', 12, 'bold'),
            bg='#3498DB',
            fg='white',
            relief='flat',
            cursor='hand2'
        )
        self.refresh_mics_btn.pack(side='right', padx=(5,0), ipady=5)
        
        # Recording section
        recording_frame = tk.Frame(content_wrapper, bg='#2C3E50')
        recording_frame.pack(fill='x', padx=20, pady=20)

        self.record_btn = tk.Button(
            recording_frame,
            text="Start Recording",
            command=self.toggle_recording,
            font=('Arial', 14, 'bold'),
            bg='#27AE60',
            fg='white',
            padx=20,
            pady=15,
            relief='flat',
            cursor='hand2',
            state='disabled',
            activebackground='#229954'
        )
        self.record_btn.pack(fill='x', pady=10)

        self.info_label = tk.Label(
            recording_frame,
            text="",
            font=('Arial', 10),
            bg='#2C3E50',
            fg='#95A5A6',
            wraplength=400
        )
        self.info_label.pack(pady=10)
        
        # Stats
        stats_frame = tk.Frame(content_wrapper, bg='#34495E')
        stats_frame.pack(fill='x', padx=20, pady=(0, 20))

        self.stats_label = tk.Label(
            stats_frame,
            text="Audio sent: 0 KB | Duration: 0s",
            font=('Arial', 10),
            bg='#34495E',
            fg='#ECF0F1',
            pady=10
        )
        self.stats_label.pack()
        self.refresh_mic_list() 

    def refresh_mic_list(self):
        """Rescan for microphones and update the dropdown."""
        self.audio_streamer.rescan_devices()
        mic_options = [device['name'] for device in self.audio_streamer.available_devices]
        
        menu = self.mic_dropdown["menu"]
        menu.delete(0, "end")

        if mic_options:
            for option in mic_options:
                menu.add_command(label=option, command=lambda value=option: self.selected_mic.set(value))
            self.selected_mic.set(mic_options[0])
            self.mic_dropdown.config(state='normal')
        else:
            self.selected_mic.set("No microphones found")
            self.mic_dropdown.config(state='disabled')
        self.update_info_label()

    def update_info_label(self):
        """Update the info label based on the system state."""
        if not PYAUDIO_AVAILABLE:
            info_text = "Warning: Install PyAudio: pip install pyaudio"
        elif not self.audio_streamer.has_devices():
            info_text = "Warning: No microphone detected. Plug one in and click Refresh."
        elif self.is_connected:
             info_text = "Click 'Start Recording' to begin."
        else:
            info_text = "Connect to a server to start recording."
        self.info_label.config(text=info_text)

    def toggle_connection(self):
        if not self.is_connected:
            self.connect_to_server()
        else:
            self.disconnect_from_server()

    def connect_to_server(self):
        server_url = self.server_entry.get().strip()
        if not server_url:
            messagebox.showerror("Error", "Please enter server address")
            return

        self.connect_btn.config(state='disabled', text="Connecting...")
        
        self.connection_thread = threading.Thread(
            target=self._connection_thread_entry,
            args=(server_url,),
            daemon=True
        )
        self.connection_thread.start()

    def _connection_thread_entry(self, server_url: str):
        """Entry point for the connection thread."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        try:
            self.event_loop.run_until_complete(self._connect_websocket(server_url))
        except Exception as e:
            logger.error(f"Connection thread error: {e}")
            self.root.after(0, self._connection_failed, str(e))
        finally:
            self.event_loop.close()

    async def _connect_websocket(self, server_url: str):
        """Connect and manage the WebSocket connection."""
        try:
            async with websockets.connect(
                server_url,
                ping_interval=20,
                ping_timeout=10,
                max_size=10_000_000
            ) as websocket:
                self.websocket = websocket
                message = await self.websocket.recv()
                data = json.loads(message)

                if data.get('type') == 'connected':
                    # FIX: Set state immediately in the background thread to prevent race condition.
                    self.is_connected = True
                    # Schedule ONLY the UI update on the main thread.
                    self.root.after(0, self._connection_success_ui)
                
                while self.is_connected:
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.root.after(0, self._connection_failed, str(e))
        finally:
            self.websocket = None
            if self.is_connected:
                self.root.after(0, self.disconnect_from_server)

    def _connection_success_ui(self):
        """Handle the UI updates for a successful connection."""
        self.status_label.config(text="Connected", fg='#27AE60')
        self.connect_btn.config(
            text="Disconnect",
            bg='#E74C3C',
            state='normal'
        )
        if PYAUDIO_AVAILABLE and self.audio_streamer.has_devices():
            self.record_btn.config(state='normal')
        self.update_info_label()
        self.server_entry.config(state='disabled')
        logger.info("✅ Connected to server")

    def _connection_failed(self, error: str):
        self.is_connected = False
        self.status_label.config(text="Connection Failed", fg='#E74C3C')
        self.connect_btn.config(text="Connect to Server", bg='#3498DB', state='normal')
        if "getaddrinfo failed" in error or "timed out" in error:
            messagebox.showerror("Connection Error", "Cannot connect to server. Please check the address and ensure the server is running.")
        else:
            messagebox.showerror("Connection Error", f"Failed to connect:\n{error}")
        logger.error(f"Connection failed: {error}")

    def disconnect_from_server(self):
        if self.is_recording:
            self.stop_recording()
        
        self.is_connected = False
        
        self.status_label.config(text="Disconnected", fg='#E74C3C')
        self.connect_btn.config(text="Connect to Server", bg='#3498DB', state='normal')
        self.record_btn.config(state='disabled')
        self.server_entry.config(state='normal')
        self.update_info_label()
        logger.info("Disconnected from server")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.is_connected:
            messagebox.showwarning("Warning", "Not connected to server")
            return

        selected_device_name = self.selected_mic.get()
        device_index = next((d['index'] for d in self.audio_streamer.available_devices if d['name'] == selected_device_name), None)

        if self.audio_streamer.start_recording(device_index=device_index):
            self.is_recording = True
            self.recording_start_time = datetime.now()
            self.bytes_sent = 0
            self.record_btn.config(text="Stop Recording", bg='#E74C3C')
            self.status_label.config(text="Recording...", fg='#E74C3C')
            
            self.connect_btn.config(state='disabled')
            self.mic_dropdown.config(state='disabled')
            self.refresh_mics_btn.config(state='disabled')
            
            asyncio.run_coroutine_threadsafe(self._send_audio_loop(), self.event_loop)
            logger.info("🎤 Recording started")

    async def _send_audio_loop(self):
        """Send audio data to the server on the correct event loop."""
        while self.is_recording:
            try:
                audio_data = self.audio_streamer.read_audio()
                if audio_data and self.websocket:
                    await self.websocket.send(audio_data)
                    self.bytes_sent += len(audio_data)
                    duration = (datetime.now() - self.recording_start_time).total_seconds()
                    kb_sent = self.bytes_sent / 1024
                    self.root.after(0, self._update_stats, kb_sent, duration)
                await asyncio.sleep(0.01)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed while sending audio.")
                self.root.after(0, self.stop_recording)
                break
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                self.root.after(0, self.stop_recording)
                break

    def _update_stats(self, kb: float, duration: float):
        """Update statistics"""
        self.stats_label.config(
            text=f"Audio sent: {kb:.1f} KB | Duration: {duration:.1f}s"
        )

    def stop_recording(self):
        self.is_recording = False
        self.audio_streamer.stop_recording()
        self.record_btn.config(text="Start Recording", bg='#27AE60')
        if self.is_connected:
            self.status_label.config(text="🟢 Connected", fg='#27AE60')
        self.update_info_label()
        self.connect_btn.config(state='normal')
        self.mic_dropdown.config(state='normal')
        self.refresh_mics_btn.config(state='normal')
        logger.info("Recording stopped")

    def on_closing(self):
        """Handle window close"""
        self.is_connected = False
        if self.event_loop and self.event_loop.is_running():
             self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        
        self.audio_streamer.cleanup()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=" * 60)
    print("AegisTranscribe Client")
    print("=" * 60)

    if not PYAUDIO_AVAILABLE:
        print("WARNING: PyAudio not installed")
        print("   Install with: pip install pyaudio")
        print("   Client will start but recording won't work")
        print()

    print("Starting client application...")

    try:
        client = ClientGUI()
        client.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
