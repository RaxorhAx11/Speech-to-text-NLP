# -*- coding: utf-8 -*-
"""
stt_engine.py
-------------
Wraps the `faster-whisper` model to provide:
- Automatic spoken-language detection (English / Hindi / Gujarati / others).
- Manual language selection (skip detection, force a language).
- Transcription with a confidence score derived from the model's
  average log-probability.
- Segment-level timestamps.

faster-whisper downloads its model weights once (requires internet the
first time) and then runs fully offline on the CPU, which is why this
approach is resilient to later network issues.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from faster_whisper import WhisperModel


# Languages this app exposes in the UI, mapped to Whisper's language codes.
# "Auto Detect" maps to None, which tells Whisper to detect the language itself.
SUPPORTED_LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
}

# Human-readable names for languages Whisper might detect automatically,
# used to display a friendly label even if the user picked "Auto Detect".
LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
}


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    detected_language: str          # e.g. "en", "hi", "gu"
    detected_language_name: str     # e.g. "English"
    confidence: float               # 0-100 percentage
    segments: List[TranscriptionSegment] = field(default_factory=list)


class STTEngine:
    """
    Loads a faster-whisper model once and reuses it for every transcription
    request (loading the model is the slow part, so we do it a single time).
    """

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8"):
        """
        model_size: "tiny", "base", "small", "medium", "large-v3".
                    "base" is a good balance of speed vs. accuracy for
                    English/Hindi/Gujarati on a CPU.
        device:     "cpu" or "cuda" if a GPU is available.
        compute_type: "int8" keeps CPU memory/speed usage low.
        """
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the speech recognition model. This usually "
                "happens the first time the app runs and needs an internet "
                "connection to download model files. Please check your "
                "network connection and try again."
            ) from exc

    def transcribe(self, audio_path: str, language_code: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe an audio file.

        language_code: a Whisper language code like "en", "hi", "gu",
                        or None to auto-detect the spoken language.
        """
        try:
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=language_code,   # None => auto-detect
                vad_filter=True,          # trims silence, improves accuracy
                beam_size=5,
            )

            segments = []
            full_text_parts = []
            logprobs = []

            for seg in segments_iter:
                segments.append(TranscriptionSegment(
                    start=seg.start, end=seg.end, text=seg.text.strip()
                ))
                full_text_parts.append(seg.text.strip())
                if seg.avg_logprob is not None:
                    logprobs.append(seg.avg_logprob)

            full_text = " ".join(full_text_parts).strip()

            # Convert average log-probability to an approximate 0-100 confidence score.
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                # avg_logprob is typically in range [-1, 0]; map it to 0-100.
                confidence = max(0.0, min(100.0, (1 + avg_logprob) * 100))
            else:
                confidence = 0.0

            detected_code = info.language or (language_code or "unknown")
            detected_name = LANGUAGE_DISPLAY_NAMES.get(detected_code, detected_code.upper())

            return TranscriptionResult(
                text=full_text,
                detected_language=detected_code,
                detected_language_name=detected_name,
                confidence=round(confidence, 1),
                segments=segments,
            )

        except FileNotFoundError as exc:
            raise RuntimeError("The recorded audio file could not be found.") from exc
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc
