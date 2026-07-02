# -*- coding: utf-8 -*-
"""
app.py
------
Main entry point: a clean, minimal Tkinter GUI for the Speech-to-Text app.

Features wired up here:
- Language selector (Auto Detect / English / Hindi / Gujarati).
- Large circular microphone button that toggles Start/Stop recording.
- Recording status indicator with a pulsing "Recording..." label.
- Scrollable, UTF-8 safe text area for the transcription
  (renders Hindi/Gujarati correctly).
- Detected language + confidence score display.
- Copy to clipboard, Save to .txt, and Clear buttons.
- Optional continuous listening mode (bonus feature).
- Timestamps shown next to each transcribed chunk.
- Friendly error dialogs for microphone and network/model errors.

Run with:  python app.py
"""

import os
import threading
import tempfile
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from audio_recorder import AudioRecorder, MicrophoneError
from stt_engine import STTEngine, SUPPORTED_LANGUAGES

# pyperclip gives cross-platform clipboard access ("Copy" button).
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Color palette - kept in one place for a clean, consistent minimal look.
# ---------------------------------------------------------------------------
COLOR_BG = "#F7F8FA"
COLOR_CARD = "#FFFFFF"
COLOR_PRIMARY = "#4F46E5"      # indigo
COLOR_PRIMARY_DARK = "#4338CA"
COLOR_DANGER = "#DC2626"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6B7280"
COLOR_BORDER = "#E5E7EB"


class SpeechToTextApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speech to Text — English / Hindi / Gujarati")
        self.root.geometry("760x640")
        self.root.minsize(620, 560)
        self.root.configure(bg=COLOR_BG)

        # --- State -----------------------------------------------------
        self.recorder = AudioRecorder()
        self.engine = None                 # Loaded lazily in a background thread
        self.is_recording = False
        self.continuous_mode = tk.BooleanVar(value=False)
        self._continuous_thread_stop = threading.Event()

        # --- Build UI ----------------------------------------------------
        self._build_ui()

        # Load the Whisper model in the background so the UI opens instantly.
        self._set_status("Loading speech model…", COLOR_MUTED)
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=6)

        container = tk.Frame(self.root, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=24, pady=20)

        # ---- Title -----------------------------------------------------
        title = tk.Label(
            container, text="🎙️ Speech to Text",
            font=("Segoe UI", 20, "bold"), bg=COLOR_BG, fg=COLOR_TEXT
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            container, text="Supports English, Hindi and Gujarati",
            font=("Segoe UI", 11), bg=COLOR_BG, fg=COLOR_MUTED
        )
        subtitle.pack(anchor="w", pady=(0, 16))

        # ---- Language selector row --------------------------------------
        lang_row = tk.Frame(container, bg=COLOR_BG)
        lang_row.pack(fill="x", pady=(0, 12))

        tk.Label(lang_row, text="Language:", font=("Segoe UI", 11),
                 bg=COLOR_BG, fg=COLOR_TEXT).pack(side="left")

        self.language_var = tk.StringVar(value="Auto Detect")
        lang_dropdown = ttk.Combobox(
            lang_row, textvariable=self.language_var,
            values=list(SUPPORTED_LANGUAGES.keys()),
            state="readonly", width=18, font=("Segoe UI", 10)
        )
        lang_dropdown.pack(side="left", padx=(10, 20))

        self.continuous_check = tk.Checkbutton(
            lang_row, text="Continuous listening mode",
            variable=self.continuous_mode, bg=COLOR_BG, fg=COLOR_TEXT,
            font=("Segoe UI", 10), activebackground=COLOR_BG,
            selectcolor=COLOR_CARD
        )
        self.continuous_check.pack(side="left")

        # ---- Mic button + status -----------------------------------------
        mic_card = tk.Frame(container, bg=COLOR_CARD, highlightbackground=COLOR_BORDER,
                             highlightthickness=1, bd=0)
        mic_card.pack(fill="x", pady=(0, 16))

        mic_inner = tk.Frame(mic_card, bg=COLOR_CARD)
        mic_inner.pack(pady=20)

        self.mic_button = tk.Button(
            mic_inner, text="🎤", font=("Segoe UI", 30),
            width=3, height=1, bg=COLOR_PRIMARY, fg="white",
            activebackground=COLOR_PRIMARY_DARK, activeforeground="white",
            relief="flat", cursor="hand2", command=self._toggle_recording
        )
        self.mic_button.pack()

        self.status_label = tk.Label(
            mic_inner, text="Idle", font=("Segoe UI", 11, "bold"),
            bg=COLOR_CARD, fg=COLOR_MUTED
        )
        self.status_label.pack(pady=(10, 0))

        # ---- Control buttons row -------------------------------------
        btn_row = tk.Frame(container, bg=COLOR_BG)
        btn_row.pack(fill="x", pady=(0, 12))

        self.start_btn = self._make_button(btn_row, "▶ Start Recording", COLOR_PRIMARY, self._start_recording)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = self._make_button(btn_row, "■ Stop Recording", COLOR_DANGER, self._stop_recording)
        self.stop_btn.pack(side="left", padx=(0, 8))
        self.stop_btn.config(state="disabled")

        self.clear_btn = self._make_button(btn_row, "🗑 Clear", COLOR_MUTED, self._clear_text)
        self.clear_btn.pack(side="left")

        # ---- Detected language / confidence -----------------------------
        info_row = tk.Frame(container, bg=COLOR_BG)
        info_row.pack(fill="x", pady=(4, 8))

        self.detected_lang_label = tk.Label(
            info_row, text="Detected language: —", font=("Segoe UI", 10),
            bg=COLOR_BG, fg=COLOR_TEXT
        )
        self.detected_lang_label.pack(side="left")

        self.confidence_label = tk.Label(
            info_row, text="Confidence: —", font=("Segoe UI", 10),
            bg=COLOR_BG, fg=COLOR_TEXT
        )
        self.confidence_label.pack(side="right")

        # ---- Transcription text area --------------------------------
        text_frame = tk.Frame(container, bg=COLOR_CARD, highlightbackground=COLOR_BORDER,
                               highlightthickness=1)
        text_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        # UTF-8 text works natively in Tkinter's Text widget on all platforms;
        # we just need a font that has Devanagari (Hindi) and Gujarati glyphs.
        self.text_area = tk.Text(
            text_frame, wrap="word", font=("Nirmala UI", 12),
            bg=COLOR_CARD, fg=COLOR_TEXT, relief="flat", padx=12, pady=12,
            yscrollcommand=scrollbar.set, undo=True
        )
        self.text_area.pack(fill="both", expand=True)
        scrollbar.config(command=self.text_area.yview)

        # ---- Bottom action row (copy / save) ---------------------------
        bottom_row = tk.Frame(container, bg=COLOR_BG)
        bottom_row.pack(fill="x", pady=(12, 0))

        self.copy_btn = self._make_button(bottom_row, "📋 Copy Text", COLOR_PRIMARY, self._copy_text)
        self.copy_btn.pack(side="left", padx=(0, 8))

        self.save_btn = self._make_button(bottom_row, "💾 Save as .txt", COLOR_PRIMARY, self._save_text)
        self.save_btn.pack(side="left")

    def _make_button(self, parent, text, color, command):
        return tk.Button(
            parent, text=text, font=("Segoe UI", 10, "bold"),
            bg=color, fg="white", activebackground=color, activeforeground="white",
            relief="flat", padx=14, pady=8, cursor="hand2", command=command
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            self.engine = STTEngine(model_size="base")
            self.root.after(0, lambda: self._set_status("Idle", COLOR_MUTED))
        except Exception as exc:
            self.root.after(0, lambda: self._set_status("Model failed to load", COLOR_DANGER))
            self.root.after(0, lambda: self._show_error("Model Load Error", str(exc)))

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.engine is None:
            self._show_error("Please wait", "The speech model is still loading. Try again in a moment.")
            return
        if self.is_recording:
            return

        try:
            self.recorder.start()
        except MicrophoneError as exc:
            self._show_error("Microphone Error", str(exc))
            return

        self.is_recording = True
        self._set_status("● Recording…", COLOR_DANGER)
        self.mic_button.config(bg=COLOR_DANGER, activebackground=COLOR_DANGER)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        if self.continuous_mode.get():
            self._continuous_thread_stop.clear()
            threading.Thread(target=self._continuous_loop, daemon=True).start()

    def _stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self._continuous_thread_stop.set()
        self.mic_button.config(bg=COLOR_PRIMARY, activebackground=COLOR_PRIMARY_DARK)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_status("Transcribing…", COLOR_MUTED)

        audio_data = self.recorder.stop()
        if audio_data.size == 0:
            self._set_status("Idle", COLOR_MUTED)
            return

        # Run transcription off the UI thread so the window never freezes.
        threading.Thread(target=self._transcribe_audio, args=(audio_data,), daemon=True).start()

    def _continuous_loop(self):
        """
        Bonus feature: continuous listening mode.
        Records in ~5 second chunks and appends each transcription with a
        timestamp, until the user presses Stop.
        """
        chunk_seconds = 5
        while not self._continuous_thread_stop.is_set():
            self._continuous_thread_stop.wait(chunk_seconds)
            if self._continuous_thread_stop.is_set():
                break
            # Briefly stop & restart the stream to flush a chunk of audio.
            audio_chunk = self.recorder.stop()
            if audio_chunk.size > 0:
                self._transcribe_audio(audio_chunk, append=True)
            if not self._continuous_thread_stop.is_set():
                try:
                    self.recorder.start()
                except MicrophoneError as exc:
                    self.root.after(0, lambda: self._show_error("Microphone Error", str(exc)))
                    break

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def _transcribe_audio(self, audio_data, append=False):
        tmp_path = None
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), "stt_recording.wav")
            self.recorder.save_wav(tmp_path, audio_data)

            language_code = SUPPORTED_LANGUAGES.get(self.language_var.get())
            result = self.engine.transcribe(tmp_path, language_code=language_code)

            self.root.after(0, lambda: self._display_result(result, append=append))
        except RuntimeError as exc:
            # Covers transcription failures and, indirectly, unexpected
            # network hiccups if the model needed to fetch extra files.
            self.root.after(0, lambda: self._show_error("Transcription Error", str(exc)))
            self.root.after(0, lambda: self._set_status("Idle", COLOR_MUTED))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _display_result(self, result, append=False):
        if not result.text:
            self._set_status("Idle" if not self.is_recording else "● Recording…", COLOR_MUTED)
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {result.text}\n"

        if append:
            self.text_area.insert("end", line)
        else:
            self.text_area.insert("end", line)

        self.text_area.see("end")

        self.detected_lang_label.config(
            text=f"Detected language: {result.detected_language_name} ({result.detected_language})"
        )
        self.confidence_label.config(text=f"Confidence: {result.confidence}%")

        self._set_status("● Recording…" if self.is_recording else "Idle",
                          COLOR_DANGER if self.is_recording else COLOR_MUTED)

    # ------------------------------------------------------------------
    # Text area actions
    # ------------------------------------------------------------------
    def _clear_text(self):
        self.text_area.delete("1.0", "end")
        self.detected_lang_label.config(text="Detected language: —")
        self.confidence_label.config(text="Confidence: —")

    def _copy_text(self):
        content = self.text_area.get("1.0", "end").strip()
        if not content:
            return
        if CLIPBOARD_AVAILABLE:
            try:
                pyperclip.copy(content)
                self._flash_status("Copied to clipboard ✓")
            except Exception:
                self._fallback_clipboard_copy(content)
        else:
            self._fallback_clipboard_copy(content)

    def _fallback_clipboard_copy(self, content):
        # Tkinter's built-in clipboard as a fallback if pyperclip is unavailable.
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._flash_status("Copied to clipboard ✓")

    def _save_text(self):
        content = self.text_area.get("1.0", "end").strip()
        if not content:
            self._show_error("Nothing to save", "The transcription box is empty.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="transcription.txt"
        )
        if not filepath:
            return

        try:
            # utf-8 encoding ensures Hindi and Gujarati text is saved correctly.
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self._flash_status("Saved ✓")
        except OSError as exc:
            self._show_error("Save Error", f"Could not save the file: {exc}")

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def _flash_status(self, message):
        previous_text = self.status_label.cget("text")
        previous_color = self.status_label.cget("fg")
        self._set_status(message, COLOR_PRIMARY)
        self.root.after(1500, lambda: self._set_status(previous_text, previous_color))

    def _show_error(self, title, message):
        messagebox.showerror(title, message)


def main():
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
