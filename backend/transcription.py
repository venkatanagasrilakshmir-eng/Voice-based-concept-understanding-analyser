"""
backend/transcription.py

Speech-to-text transcription using OpenAI Whisper.
Loads the model once (lazy singleton) so repeated calls are fast.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import List, Dict, Any

from config import settings


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "word_count": self.word_count,
            "segments": self.segments,
        }


@functools.lru_cache(maxsize=1)
def _load_model(model_size: str):
    """Load and cache the Whisper model (expensive, done once per process)."""
    import whisper  # imported lazily so the rest of the app works without it installed
    return whisper.load_model(model_size)


class Transcriber:
    """Wraps OpenAI Whisper for spoken-audio-to-text transcription."""

    def __init__(self, model_size: str | None = None):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: path to a .wav/.mp3/.m4a file
            language: optional ISO language hint (e.g. "en"); auto-detected if None

        Returns:
            TranscriptionResult
        """
        result = self.model.transcribe(audio_path, language=language, fp16=False)

        text = result.get("text", "").strip()
        segments = [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
            for seg in result.get("segments", [])
        ]
        duration = segments[-1]["end"] if segments else 0.0

        return TranscriptionResult(
            text=text,
            language=result.get("language", "unknown"),
            duration=duration,
            segments=segments,
            word_count=len(text.split()),
        )
