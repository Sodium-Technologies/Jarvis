""""Hey Jarvis" wake word, using openWakeWord's free pre-trained model.

How it works:
  1. Audio is checked in 80 ms pieces by a small wake-word model. Nothing is
     stored or sent anywhere; audio not followed by the wake word is discarded.
  2. When the model fires, Jarvis records your request until you pause.
  3. Whisper transcribes the wake phrase *and* the request together. If it did
     not actually hear "Jarvis", the trigger is ignored (a cheap second check
     against false alarms such as "hey Charlie").
"""

from __future__ import annotations

import glob
import os
import re
from collections import deque
from typing import Callable

import numpy as np

from .listen import FRAME, SAMPLE_RATE, EndpointSettings, Endpointer

CHUNK = 1280  # openWakeWord's 80 ms input size


# ---- the wake-word model ------------------------------------------------------


def find_model(name: str = "hey_jarvis") -> str:
    """Path to the ONNX wake-word model, downloading it once if needed."""
    try:
        import openwakeword
    except ImportError as exc:
        raise RuntimeError("Wake word support is not installed. Run: pip install -e \".[voice]\"") from exc
    folder = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    found = glob.glob(os.path.join(folder, f"{name}*.onnx"))
    if not found and hasattr(openwakeword, "utils") and hasattr(openwakeword.utils, "download_models"):
        openwakeword.utils.download_models(model_names=[name])  # one-time download (~2 MB)
        found = glob.glob(os.path.join(folder, f"{name}*.onnx"))
    if not found:
        raise RuntimeError(f"Wake word model '{name}' not found in {folder}")
    return found[0]


class OpenWakeWordScorer:
    """Scores 80 ms chunks of 16 kHz int16 audio: 0 (no) to 1 (definitely "hey Jarvis")."""

    def __init__(self, name: str = "hey_jarvis"):
        import inspect
        import warnings

        from openwakeword.model import Model

        path = find_model(name)
        params = inspect.signature(Model.__init__).parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. "CUDA provider not available": irrelevant on a Mac
            if "wakeword_models" in params:  # openWakeWord 0.5+
                self.model = Model(wakeword_models=[path], inference_framework="onnx")
            else:                            # 0.4
                self.model = Model(wakeword_model_paths=[path])

    def __call__(self, chunk: np.ndarray) -> float:
        return float(max(self.model.predict(chunk).values()))

    def reset(self) -> None:
        self.model.reset()


class WakeDetector:
    def __init__(self, scorer, threshold: float = 0.5):
        self.scorer = scorer
        self.threshold = threshold
        self._buf = np.zeros(0, np.float32)
        self.last_score = 0.0

    def feed(self, frame: np.ndarray) -> bool:
        """Add float32 audio; True when the wake word was just heard."""
        self._buf = np.concatenate([self._buf, frame])
        fired = False
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            self.last_score = self.scorer(pcm)
            if self.last_score >= self.threshold:
                fired = True
        if fired:
            self.reset()
        return fired

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        if hasattr(self.scorer, "reset"):
            self.scorer.reset()


# ---- confirming and removing the wake phrase from the transcript ----------------

# Whisper's usual spellings of "Jarvis". "Travis", "service" etc. are deliberately
# not accepted: hearing those is exactly the false alarm this check is for.
_WAKE = re.compile(r"\b(?:jarvis|jarvi|jervis|javis|jarvus|jarviss)\b", re.I)
_FILLER = re.compile(r"^\W*(?:(?:uh|um|so|and)\W+)*$", re.I)
_GREETING = re.compile(r"^\W*(?:(?:uh|um|so|and)\W+)*(?:hey|hi|hay|hello|ok(?:ay)?|yo)\W+$", re.I)


def split_wake(text: str) -> tuple[bool, str]:
    """(was Jarvis being addressed?, the request after the wake phrase).

    Addressed: "Hey Jarvis, ...", "OK Jarvis ...", or "Jarvis, ..." / "Jarvis." /
    just "Jarvis". Not addressed: "Jarvis is the name of ...", "Hey Charlie".
    """
    m = _WAKE.search(text)
    if not m or m.start() > 25:
        return False, text.strip()
    lead, after = text[: m.start()], text[m.end():]
    greeted = bool(_GREETING.match(lead))
    vocative = bool(_FILLER.match(lead)) and bool(re.match(r"\s*([,.!?:;-]|$)", after))
    if not (greeted or vocative):
        return False, text.strip()
    return True, re.sub(r"^[\s,.!?:;-]+", "", after).strip()


# ---- the always-on listener ---------------------------------------------------


class WakeListener:
    """Waits for "hey Jarvis", then records the request that follows."""

    def __init__(self, detector: WakeDetector, settings: EndpointSettings | None = None,
                 preroll_s: float = 2.0):
        self.detector = detector
        self.settings = settings or EndpointSettings(no_speech_timeout_s=6.0)
        self.ring: deque[np.ndarray] = deque(maxlen=int(preroll_s * SAMPLE_RATE / FRAME))
        self.levels: deque[float] = deque(maxlen=int(3 * SAMPLE_RATE / FRAME))

    def noise_threshold(self) -> float:
        s = self.settings
        floor = float(np.percentile(self.levels, 30)) if self.levels else 0.0
        return max(s.min_threshold, floor * s.noise_factor)

    def wait_for_request(self, read: Callable[[], np.ndarray],
                         paused: Callable[[], bool] = lambda: False,
                         on_wake: Callable[[], None] = lambda: None) -> np.ndarray | None:
        """Block until the wake word, then return wake phrase + request audio.

        Returns None if the wake word was heard but nothing followed in time.
        While paused() is true (Jarvis is talking), audio is discarded so Jarvis
        never wakes itself up.
        """
        while True:
            frame = read()
            if paused():
                self.ring.clear()
                self.detector.reset()
                continue
            self.ring.append(frame)
            self.levels.append(Endpointer.rms(frame))
            if self.detector.feed(frame):
                break

        wake_audio = np.concatenate(list(self.ring))
        self.ring.clear()
        on_wake()
        ep = Endpointer(self.settings, threshold=self.noise_threshold())
        while not ep.feed(read()):
            pass
        if ep.result is None:
            return wake_audio  # just "hey Jarvis"; the transcript check decides what to do
        gap = np.zeros(int(0.2 * SAMPLE_RATE), np.float32)
        return np.concatenate([wake_audio, gap, ep.result])
