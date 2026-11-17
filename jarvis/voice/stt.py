"""Speech to text with Whisper (faster-whisper), fully offline after the first
model download."""

from __future__ import annotations

import re

import numpy as np

# Whisper sometimes "hears" these in near-silence; drop them when the clip is short.
_PHANTOM = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "."}


class Transcriber:
    def __init__(self, model: str = "small.en", language: str = "en"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Speech recognition is not installed. Run: pip install -e \".[voice]\"") from exc
        # int8 on the CPU is fast on Apple Silicon and needs no GPU setup.
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        # English-only models (*.en) need "en"; for multilingual models "" means auto-detect
        # (e.g. speaking Urdu and English).
        self.language = "en" if model.endswith(".en") else (language or None)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio.astype(np.float32), language=self.language, beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return clean_transcript(text, seconds=len(audio) / 16_000)


def clean_transcript(text: str, seconds: float) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"\w", text):
        return ""
    if seconds < 2.0 and text.lower() in _PHANTOM:
        return ""
    return text
