import threading
import time

import numpy as np
import pytest

from jarvis.agent import Agent
from jarvis.config import load_config
from jarvis.llm import OllamaClient
from jarvis.memory import Memory
from jarvis.tools import Tool, ToolRegistry
from jarvis.voice.listen import FRAME, SAMPLE_RATE, EndpointSettings, Endpointer
from jarvis.voice.session import EXIT_PHRASES, VoiceSession
from jarvis.voice.speak import SentenceStreamer, Speaker, clean_for_speech, speech_command
from jarvis.voice.stt import clean_transcript

rng = np.random.default_rng(0)


def noise(seconds, level=0.002):
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.float32)


def voice(seconds, level=0.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 220 * t) * level * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def run_endpointer(audio, **kw):
    ep = Endpointer(EndpointSettings(**kw))
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        if ep.feed(audio[i : i + FRAME]):
            break
    return ep


# ---- end-of-speech detection ------------------------------------------------


def test_endpointer_captures_speech_and_stops_on_pause():
    audio = np.concatenate([noise(0.6), voice(1.5), noise(0.3), voice(0.8), noise(3.0)])
    ep = run_endpointer(audio)
    assert ep.done and ep.result is not None
    secs = len(ep.result) / SAMPLE_RATE
    # Both phrases (the short 0.3 s gap does not end it), plus pre-roll and a short tail.
    assert 2.6 < secs < 3.3
    # Stopped after ~0.9 s of silence, not at the end of the 3 s of noise.
    consumed = len(ep.frames) * FRAME / SAMPLE_RATE
    assert consumed < 0.6 + 1.5 + 0.3 + 0.8 + 1.2


def test_endpointer_times_out_when_nobody_speaks():
    ep = run_endpointer(noise(10), no_speech_timeout_s=3)
    assert ep.done and ep.result is None
    assert len(ep.frames) * FRAME / SAMPLE_RATE == pytest.approx(3, abs=0.1)


def test_endpointer_adapts_to_noisy_room_and_caps_length():
    loud_room = noise(12, level=0.03)  # fan / street noise well above the floor
    assert run_endpointer(loud_room, no_speech_timeout_s=2).result is None
    ep = run_endpointer(np.concatenate([noise(0.4), voice(20)]), max_utterance_s=5)
    assert ep.result is not None and len(ep.result) / SAMPLE_RATE < 6


# ---- transcript and speech text cleanup ----------------------------------------


def test_clean_transcript_drops_whisper_phantoms():
    assert clean_transcript("  Thank you. ", seconds=0.8) == ""
    assert clean_transcript("Thank you.", seconds=4) == "Thank you."
    assert clean_transcript(" ... ", seconds=3) == ""
    assert clean_transcript("What's  the rent\nfor 4B?", 3) == "What's the rent for 4B?"


def test_clean_for_speech():
    md = ("## Summary\n- **Rent:** $2,150 per month\n- See `Lease_Unit_4B.pdf` at "
          "/Users/naveed/Documents/LUXE/Lease_Unit_4B_2025.pdf\n"
          "```python\nprint(1)\n```\nMore at https://example.com/x | done")
    out = clean_for_speech(md)
    assert "#" not in out and "**" not in out and "`" not in out and "|" not in out
    assert "Rent: $2,150 per month" in out
    assert "Lease Unit 4B 2025" in out and "/Users" not in out
    assert "print(1)" not in out and "code on screen" in out
    assert "the link on screen" in out


def test_sentence_streamer_splits_naturally():
    s = SentenceStreamer()
    text = ("Dr. Patel paid $2,150.00 on Oct. 1 for the unit at 12 Elm St. in Baltimore. "
            "The deposit is held in escrow. Ok. Anything else?")
    spoken = []
    for i in range(0, len(text), 3):  # stream in small pieces like a model does
        spoken += s.feed(text[i : i + 3])
    spoken += s.flush()
    assert spoken == [
        "Dr. Patel paid $2,150.00 on Oct. 1 for the unit at 12 Elm St. in Baltimore.",
        "The deposit is held in escrow.",
        "Ok. Anything else?",  # very short sentences are joined to the next one
    ]


def test_sentence_streamer_waits_for_code_blocks():
    s = SentenceStreamer()
    out = s.feed("Here is the script. It works fine.\n```\nx = 1. y = 2.\n")
    assert out == ["Here is the script. It works fine."]
    assert s.feed("z = 3.\n") == []
    out = s.feed("```\nRun it when ready. ") + s.flush()
    assert out == ["I've put the code on screen. Run it when ready."]


# ---- speaking -----------------------------------------------------------------


def test_speech_command_for_mac_and_missing_voice(monkeypatch):
    assert speech_command("Daniel", 180, platform="darwin", voices=["Daniel", "Samantha"]) == \
        ["say", "-v", "Daniel", "-r", "180", "-f", "-"]
    # A voice that isn't installed falls back to the system default instead of failing.
    assert speech_command("Jamie (Premium)", 0, platform="darwin", voices=["Daniel"]) == ["say", "-f", "-"]
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert speech_command("Daniel", 0, platform="linux") is None


class FakeProc:
    def __init__(self, log, text, duration):
        self.log, self.text = log, text
        self.done = threading.Event()
        self.killed = False
        threading.Timer(duration, self.done.set).start()

    def wait(self):
        self.done.wait()
        self.log.append(("done" if not self.killed else "stopped", self.text))

    def poll(self):
        return 0 if self.done.is_set() else None

    def terminate(self):
        self.killed = True
        self.done.set()


def fake_speaker(duration=0.05):
    log = []
    sp = Speaker(command=["say", "-f", "-"], runner=lambda cmd, text: FakeProc(log, text, duration))
    return sp, log


def test_speaker_plays_in_order_and_can_be_interrupted():
    sp, log = fake_speaker(0.05)
    for t in ("One.", "Two.", "Three."):
        sp.say(t)
    assert sp.busy()
    assert sp.wait(2)
    assert log == [("done", "One."), ("done", "Two."), ("done", "Three.")]

    log.clear()
    sp2, log2 = fake_speaker(5)  # long sentence
    sp2.say("A very long answer.")
    sp2.say("Never spoken.")
    time.sleep(0.1)
    sp2.stop()
    assert sp2.wait(2)
    assert log2 == [("stopped", "A very long answer.")]


def test_disabled_speaker_is_silent():
    sp = Speaker(command=None)
    sp.say("hello")
    assert not sp.enabled and not sp.busy() and sp.wait(0.1)


# ---- the talk loop ------------------------------------------------------------


class FakeMic:
    def __init__(self, clips):
        self.clips = list(clips)

    def record(self):
        return self.clips.pop(0)


class FakeWhisper:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio):
        return self.texts.pop(0)


class NullPrinter:
    def __call__(self, kind, text):
        pass

    def finish(self):
        pass


def make_session(env, mic_clips, heard, typed, tools=None, confirm=lambda n, a: False, hands_free=False):
    cfg = load_config()
    agent = Agent(cfg, OllamaClient(cfg.ollama_host), Memory(cfg.db_path), tools=tools)
    sp, log = fake_speaker(0.01)
    lines = []
    typed = iter(typed)
    # A real user hears the reply before typing again (typing early would interrupt it).
    wait_then_type = lambda prompt: (sp.wait(2), next(typed))[1]
    session = VoiceSession(
        agent, FakeMic(mic_clips), FakeWhisper(heard), sp,
        printer_factory=NullPrinter, on_tool=lambda n, a: None, confirm=confirm,
        input_fn=wait_then_type, out=lines.append, hands_free=hands_free, sound=lambda: None,
    )
    return session, log, lines


def test_voice_turn_speaks_answer_but_not_reasoning(jarvis_env):
    jarvis_env.queue_text("The rent for unit 4B is $2,150. It is due on the first.",
                          thinking="SECRET REASONING should never be spoken.")
    session, log, lines = make_session(
        jarvis_env, [voice(1)], ["What's the rent for 4B?"], ["", "/exit"])
    assert session.run() == 0
    session.speaker.wait(2)
    spoken = " ".join(t for _, t in log)
    assert "you> What's the rent for 4B?" in lines
    assert jarvis_env.requests[0]["messages"][-1]["content"] == "What's the rent for 4B?"
    assert "$2,150" in spoken and "SECRET" not in spoken


def test_typing_still_works_and_silence_is_ignored(jarvis_env):
    jarvis_env.queue_text("Typed answer.")
    session, log, lines = make_session(jarvis_env, [None], [], ["", "hello there", "/exit"])
    session.run()
    assert "(I didn't hear anything.)" in lines
    assert len(jarvis_env.requests) == 1 and jarvis_env.requests[0]["messages"][-1]["content"] == "hello there"


def test_goodbye_by_voice_ends_session(jarvis_env):
    session, log, lines = make_session(jarvis_env, [voice(1)], ["Goodbye, Jarvis."], [""])
    assert session.run() == 0
    assert jarvis_env.requests == [] and ("done", "Goodbye.") in log
    assert EXIT_PHRASES.match("stop listening") and not EXIT_PHRASES.match("goodbye email to Doug")


def test_confirmation_is_typed_not_spoken(jarvis_env):
    ran = []
    reg = ToolRegistry()
    reg.register(Tool("delete_note", "x", {"type": "object", "properties": {}}, lambda: ran.append(1) or "ok",
                      confirm=True))
    jarvis_env.queue_tool_call("delete_note", {})
    jarvis_env.queue_text("Okay, I left it.")
    asked = []
    session, log, lines = make_session(
        jarvis_env, [voice(1)], ["yes delete it"], ["", "/exit"], tools=reg,
        confirm=lambda n, a: asked.append(n) or False)
    session.run()
    session.speaker.wait(2)
    assert asked == ["delete_note"] and ran == []
    assert any("confirmation on screen" in t for _, t in log)


def test_hands_free_listens_again_after_reply(jarvis_env):
    jarvis_env.queue_text("First answer.")
    jarvis_env.queue_text("Second answer.")
    session, log, lines = make_session(
        jarvis_env, [voice(1), voice(1), voice(1)], ["one", "two", "bye"], [""], hands_free=True)
    assert session.run() == 0
    assert [r["messages"][-1]["content"] for r in jarvis_env.requests] == ["one", "two"]
