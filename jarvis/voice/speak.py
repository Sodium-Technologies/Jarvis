"""Text to speech with the computer's built-in voices.

macOS: the `say` command (free, offline). Install nicer voices in System
Settings → Accessibility → Spoken Content → System voice → Manage Voices.
Linux: espeak-ng / espeak if installed. Otherwise speech is skipped.

Replies are spoken sentence by sentence while the model is still writing, so
Jarvis starts talking quickly instead of after the whole answer.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable

# ---- making text speakable --------------------------------------------------

_CODE_BLOCK = re.compile(r"```.*?(```|$)", re.S)
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:~|/)[\w\-. ]*(?:/[\w\-. ]+)+\.\w{1,5}")


def clean_for_speech(text: str) -> str:
    """Strip markdown and other things that sound bad read aloud."""
    text = _CODE_BLOCK.sub(" I've put the code on screen. ", text)
    text = _URL.sub("the link on screen", text)
    text = _PATH.sub(lambda m: Path(m.group(0)).stem.replace("_", " ").replace("-", " "), text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+", "", text, flags=re.M)  # headings, quotes, bullets
    text = re.sub(r"(\*\*|__|\*|_)(\S.*?\S|\S)\1", r"\2", text)                  # bold / italics
    text = re.sub(r"^\s*\|?[-:| ]{3,}\|?\s*$", "", text, flags=re.M)              # table rules
    text = text.replace("|", ", ")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)                          # [label](link)
    return re.sub(r"\s+", " ", text).strip()


_ABBREV = ("mr", "mrs", "ms", "dr", "st", "jr", "sr", "inc", "llc", "ltd", "co", "vs", "etc",
           "e.g", "i.e", "approx", "apt", "no", "jan", "feb", "mar", "apr", "jun", "jul",
           "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "a.m", "p.m")


class SentenceStreamer:
    """Collects streamed tokens and releases complete sentences."""

    _END = re.compile(r"([.!?]+[\"')\]]*|:)(\s+)|\n\s*\n")

    def __init__(self, min_chars: int = 25):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, token: str) -> list[str]:
        self.buf += token
        out = []
        while True:
            # Never cut inside a code block (it is replaced by a short phrase later),
            # but do release complete sentences that come before it.
            limit = len(self.buf)
            if self.buf.count("```") % 2 == 1:
                limit = self.buf.rindex("```")
            m = None
            for cand in self._END.finditer(self.buf, 0, limit):
                end = cand.end()
                head = self.buf[: cand.start()].rstrip()
                last_word = re.findall(r"[\w.]+$", head)
                if cand.group(1) and cand.group(1).startswith(".") and last_word and \
                        last_word[0].lower().rstrip(".") in _ABBREV:
                    continue  # "Dr. Smith", "Elm St. unit"
                if len(self.buf[:end].strip()) < self.min_chars and cand.group(0).strip() != "":
                    continue  # too short to bother; join with the next sentence
                m = cand
                break
            if not m:
                return out
            sentence, self.buf = self.buf[: m.end()], self.buf[m.end():]
            spoken = clean_for_speech(sentence)
            if spoken:
                out.append(spoken)

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        spoken = clean_for_speech(rest)
        return [spoken] if spoken else []


# ---- speaking ---------------------------------------------------------------

Runner = Callable[[list[str], str], subprocess.Popen]


def _popen(cmd: list[str], text: str) -> subprocess.Popen:
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.stdin.write(text.encode("utf-8"))
    p.stdin.close()
    return p


def available_voices() -> list[str]:
    if sys.platform != "darwin" or not shutil.which("say"):
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    # Lines look like: "Daniel              en_GB    # Hello! My name is Daniel."
    return [re.split(r"\s{2,}", line.strip())[0] for line in out.splitlines() if line.strip()]


def speech_command(voice: str = "", rate: int = 0, platform: str = sys.platform,
                   voices: list[str] | None = None) -> list[str] | None:
    """The command that reads text from stdin aloud, or None if no engine exists."""
    if platform == "darwin" and (shutil.which("say") or voices is not None):
        cmd = ["say"]
        voices = available_voices() if voices is None else voices
        if voice and voice in voices:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        return cmd + ["-f", "-"]  # text via stdin: no argument-escaping issues
    for engine in ("espeak-ng", "espeak"):
        if shutil.which(engine):
            return [engine] + (["-s", str(rate)] if rate else []) + ["--stdin"]
    return None


class Speaker:
    """Speaks queued sentences one after another on a background thread."""

    def __init__(self, voice: str = "Daniel", rate: int = 0, *, command: list[str] | None = None,
                 runner: Runner = _popen):
        self.command = command if command is not None else speech_command(voice, rate)
        self.runner = runner
        self._items: deque[str] = deque()
        self._proc: subprocess.Popen | None = None
        self._speaking = False
        self._closed = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.command is not None

    def say(self, text: str) -> None:
        if self.enabled and text.strip():
            with self._cv:
                self._items.append(text)
                self._cv.notify_all()

    def _idle(self) -> bool:
        return not self._items and not self._speaking

    def busy(self) -> bool:
        with self._cv:
            return not self._idle()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or stopped)."""
        with self._cv:
            return self._cv.wait_for(self._idle, timeout)

    def stop(self) -> None:
        """Stop talking now and drop anything queued (e.g. the user interrupted)."""
        with self._cv:
            self._items.clear()
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()

    def close(self) -> None:
        self.stop()
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._items or self._closed)
                if self._closed:
                    return
                text = self._items.popleft()
                # Marked as speaking before the lock is released, so wait()/busy()
                # never see a gap between taking a sentence and starting to say it.
                self._speaking = True
                try:
                    self._proc = self.runner(self.command, text)
                except OSError:
                    self._proc = None
            try:
                if self._proc is not None:
                    self._proc.wait()
            finally:
                with self._cv:
                    self._proc = None
                    self._speaking = False
                    self._cv.notify_all()
