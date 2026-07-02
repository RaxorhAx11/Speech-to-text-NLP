# -*- coding: utf-8 -*-
"""
audio_recorder.py
------------------
Handles microphone recording using the `sounddevice` library.

Responsibilities:
- Open a microphone input stream.
- Continuously capture audio chunks into memory while recording is active.
- Save the captured audio to a WAV file for the STT engine to consume.
- Raise clear, catchable exceptions when the microphone is unavailable
  or permission is denied, so the UI layer can show a friendly message.
"""

import queue
import wave
import numpy as np
import sounddevice as sd


class MicrophoneError(Exception):
    """Raised when the microphone cannot be opened (permission / no device)."""
    pass


class AudioRecorder:
    """
    Records audio from the default microphone in a background-friendly way.

    Usage:
        recorder = AudioRecorder()
        recorder.start()
        ... user speaks ...
        recorder.stop()
        recorder.save_wav("output.wav")
    """

    def __init__(self, samplerate: int = 16000, channels: int = 1):
        # 16kHz mono is the format Whisper expects internally, so we
        # record directly at that rate to avoid an extra resampling step.
        self.samplerate = samplerate
        self.channels = channels
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._recording = False
        self._frames = []

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice for every audio block captured."""
        if status:
            # Non-fatal warnings (e.g. buffer overflow) are printed, not raised.
            print(f"[AudioRecorder] Stream status: {status}")
        # Copy is required because sounddevice reuses the buffer.
        self._audio_queue.put(indata.copy())

    def start(self):
        """Start capturing audio from the microphone."""
        if self._recording:
            return  # already recording, ignore duplicate calls

        self._frames = []
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
        except Exception as exc:
            # Covers: no microphone found, permission denied, device busy, etc.
            raise MicrophoneError(
                "Could not access the microphone. Please check that a "
                "microphone is connected and that this application has "
                "permission to use it."
            ) from exc

    def stop(self) -> np.ndarray:
        """Stop capturing and return the recorded audio as a numpy array."""
        if not self._recording:
            return np.array([], dtype="int16")

        self._recording = False
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

        # Drain the queue into a single contiguous array.
        while not self._audio_queue.empty():
            self._frames.append(self._audio_queue.get())

        if not self._frames:
            return np.array([], dtype="int16")

        return np.concatenate(self._frames, axis=0)

    def save_wav(self, filepath: str, audio_data: np.ndarray):
        """Persist recorded audio data to a 16-bit PCM WAV file."""
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 -> 2 bytes per sample
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_data.tobytes())

    @property
    def is_recording(self) -> bool:
        return self._recording
