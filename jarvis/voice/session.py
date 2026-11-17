"""The talk loop: listen → transcribe → think → speak, with typing as a fallback."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from ..llm import LLMError
from .speak import SentenceStreamer

EXIT_PHRASES = re.compile(r"^\W*(goodbye|bye|stop listening|exit|quit)(,?\s*jarvis)?\W*$", re.I)
LISTEN_SOUND = Path("/System/Library/Sounds/Tink.aiff")


def chime() -> None:
    """A short 'I'm listening' sound on macOS (does nothing elsewhere)."""
    if sys.platform == "darwin" and LISTEN_SOUND.exists():
        subprocess.Popen(["afplay", str(LISTEN_SOUND)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class VoiceSession:
    def __init__(self, agent, mic, transcriber, speaker, *,
                 printer_factory: Callable, on_tool: Callable, confirm: Callable,
                 input_fn: Callable[[str], str] = input, out: Callable[[str], None] = print,
                 hands_free: bool = False, sound: Callable[[], None] = chime):
        self.agent, self.mic, self.speaker = agent, mic, speaker
        self._transcriber = transcriber  # object, or a zero-arg function that loads it
        self.printer_factory, self.on_tool, self._confirm = printer_factory, on_tool, confirm
        self.input, self.out, self.hands_free, self.sound = input_fn, out, hands_free, sound

    @property
    def transcriber(self):
        if callable(self._transcriber) and not hasattr(self._transcriber, "transcribe"):
            self._transcriber = self._transcriber()
        return self._transcriber

    def confirm(self, name: str, args: dict) -> bool:
        # A misheard "yes" must never approve an action, so confirmation is always typed.
        self.speaker.say("I need your confirmation on screen.")
        return self._confirm(name, args)

    def listen(self) -> str:
        self.sound()
        self.out("🎙  Listening… (pause when you're done)")
        audio = self.mic.record()
        if audio is None:
            self.out("(I didn't hear anything.)")
            return ""
        self.out("…transcribing")
        return self.transcriber.transcribe(audio)

    def respond(self, text: str) -> None:
        printer = self.printer_factory()
        streamer = SentenceStreamer()

        def on_token(kind: str, token: str) -> None:
            printer(kind, token)
            if kind == "content":  # reasoning is shown (if enabled) but never spoken
                for sentence in streamer.feed(token):
                    self.speaker.say(sentence)

        try:
            self.agent.ask(text, on_token=on_token, on_tool=self.on_tool, confirm=self.confirm)
        except LLMError as exc:
            printer.finish()
            self.out(f"Error: {exc}")
            self.speaker.say("Sorry, something went wrong. The details are on screen.")
            return
        for sentence in streamer.flush():
            self.speaker.say(sentence)
        printer.finish()

    def run(self) -> int:
        first = True
        while True:
            try:
                if self.hands_free and not first:
                    self.speaker.wait()  # don't listen to our own voice
                    typed = ""
                else:
                    typed = self.input("\n[Enter] talk · or type · /exit: ").strip()
                first = False
                self.speaker.stop()  # pressing Enter while Jarvis talks interrupts it
                if typed in ("/exit", "/quit"):
                    return 0
                text = typed or self.listen()
                if not text:
                    continue
                if not typed:
                    self.out(f"you> {text}")
                    if EXIT_PHRASES.match(text):
                        self.speaker.say("Goodbye.")
                        self.speaker.wait(5)
                        return 0
                self.out("jarvis>")
                self.respond(text)
            except (EOFError, KeyboardInterrupt):
                self.speaker.stop()
                self.out("")
                return 0
