"""Microphone capture with automatic end-of-speech detection.

The Endpointer is pure logic (feed it audio frames, it decides when speech
started and ended), so it is tested without a microphone. The Microphone
class wraps `sounddevice`, which is only imported when actually recording.
"""

from __future__ import annotations

import queue
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000          # Whisper expects 16 kHz mono
FRAME = 480                   # 30 ms frames


@dataclass
class EndpointSettings:
    silence_ms: int = 900         # this much quiet after speech ends the utterance
    start_frames: int = 3         # ~90 ms of loud frames to count as speech starting
    preroll_ms: int = 300         # keep a little audio before speech started
    no_speech_timeout_s: float = 8.0
    max_utterance_s: float = 45.0
    calibration_frames: int = 10  # first 300 ms measure the room's background noise
    min_threshold: float = 0.008  # RMS floor so a silent room isn't "speech"
    noise_factor: float = 3.0


class Endpointer:
    """Decides, frame by frame, when an utterance starts and ends."""

    def __init__(self, settings: EndpointSettings | None = None, threshold: float | None = None):
        """threshold: a known speech threshold (skips measuring room noise first,
        e.g. right after the wake word, when the user may already be talking)."""
        self.s = settings or EndpointSettings()
        self.preset = threshold
        self.reset()

    def reset(self) -> None:
        self.frames: list[np.ndarray] = []
        self.noise: list[float] = []
        self.threshold = self.preset if self.preset is not None else self.s.min_threshold
        self.started_at: int | None = None
        self.loud_run = 0
        self.quiet_run = 0
        self.done = False
        self.result: np.ndarray | None = None

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0

    def feed(self, frame: np.ndarray) -> bool:
        """Add one frame. Returns True when the utterance is finished (or timed out)."""
        if self.done:
            return True
        self.frames.append(frame)
        n = len(self.frames)
        level = self.rms(frame)
        ms = lambda count: count * FRAME * 1000 // SAMPLE_RATE

        if self.preset is None and self.started_at is None and n <= self.s.calibration_frames:
            self.noise.append(level)
            self.threshold = max(self.s.min_threshold, float(np.median(self.noise)) * self.s.noise_factor)
            return False

        loud = level >= self.threshold
        if self.started_at is None:
            self.loud_run = self.loud_run + 1 if loud else 0
            if self.loud_run >= self.s.start_frames:
                self.started_at = n - self.loud_run
            elif ms(n) / 1000 >= self.s.no_speech_timeout_s:
                return self._finish(None)
            return False

        self.quiet_run = 0 if loud else self.quiet_run + 1
        if ms(self.quiet_run) >= self.s.silence_ms:
            return self._finish(n - self.quiet_run)
        if ms(n - self.started_at) / 1000 >= self.s.max_utterance_s:
            return self._finish(n)
        return False

    def _finish(self, end: int | None) -> bool:
        self.done = True
        if end is not None and self.started_at is not None:
            pre = self.s.preroll_ms * SAMPLE_RATE // 1000 // FRAME
            tail = 6  # ~180 ms after the last loud frame, so word endings aren't clipped
            start = max(0, self.started_at - pre)
            self.result = np.concatenate(self.frames[start : min(len(self.frames), end + tail)])
        return True


class Microphone:
    """Records one utterance from the default input device."""

    def __init__(self, settings: EndpointSettings | None = None):
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Microphone support is not installed. Run: pip install -e \".[voice]\""
            ) from exc
        self.settings = settings or EndpointSettings()

    @contextmanager
    def open(self):
        """Keep the microphone open; yields a stream with read() and drain()."""
        import sounddevice as sd

        stream = FrameStream()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME, callback=stream.callback):
            yield stream

    def record(self) -> np.ndarray | None:
        """Block until the user finishes speaking. None if nothing was said."""
        ep = Endpointer(self.settings)
        with self.open() as stream:
            while not ep.feed(stream.read()):
                pass
        return ep.result


class FrameStream:
    """Microphone frames handed from the audio thread to the main thread."""

    def __init__(self):
        self.q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(self, indata, _frames, _time, _status) -> None:
        self.q.put(indata[:, 0].copy())

    def read(self) -> np.ndarray:
        while True:  # short timeouts keep Ctrl+C responsive
            try:
                return self.q.get(timeout=0.5)
            except queue.Empty:
                continue

    def drain(self) -> None:
        """Throw away audio that piled up while Jarvis was busy thinking."""
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
