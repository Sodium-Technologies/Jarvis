"""Speech to text with Whisper (faster-whisper), fully offline after the first
model download."""

from __future__ import annotations

import logging
import os
import re
import warnings

import numpy as np

# Whisper sometimes "hears" these in near-silence; drop them when the clip is short.
_PHANTOM = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "."}


def load_whisper(model: str):
    """Load a Whisper model, using the downloaded copy when there is one.

    Only the very first run contacts Hugging Face (to download the model);
    after that Jarvis starts offline and makes no network request at all.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Speech recognition is not installed. Run: pip install -e \".[voice]\"") from exc
    # int8 on the CPU is fast on Apple Silicon and needs no GPU setup.
    try:
        return WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        pass  # not downloaded yet
    # First run: download anonymously. No Hugging Face account or token is needed;
    # silence the "set a HF_TOKEN" notice, which only concerns download speed limits.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WhisperModel(model, device="cpu", compute_type="int8")


class Transcriber:
    def __init__(self, model: str = "small.en", language: str = "en", loader=load_whisper,
                 vocabulary: str = "Jarvis"):
        self.model = loader(model)
        # Words Whisper should expect (names, jargon), e.g. "Jarvis, Reigz, Buildium".
        self.vocabulary = vocabulary.strip() or None
        # English-only models (*.en) need "en"; for multilingual models "" means auto-detect
        # (e.g. speaking Urdu and English).
        self.language = "en" if model.endswith(".en") else (language or None)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio.astype(np.float32), language=self.language, beam_size=5,
            vad_filter=True, condition_on_previous_text=False, hotwords=self.vocabulary,
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
