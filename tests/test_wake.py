import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pytest

from jarvis.voice.listen import FRAME, SAMPLE_RATE, EndpointSettings
from jarvis.voice.session import VoiceSession
from jarvis.voice.wake import WakeDetector, WakeListener, split_wake
from test_voice import NullPrinter, FakeWhisper, fake_speaker, noise, voice


class EndOfAudio(Exception):
    pass


def reader(audio):
    frames = [audio[i : i + FRAME] for i in range(0, len(audio) - FRAME + 1, FRAME)]
    it = iter(frames)

    def read():
        try:
            return next(it)
        except StopIteration:
            raise EndOfAudio

    return read


# ---- transcript check -----------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Hey Jarvis, what's the rent for 4B?", (True, "what's the rent for 4B?")),
    ("hey jarvis", (True, "")),
    ("Hey, Jarvis. Any emails from Doug?", (True, "Any emails from Doug?")),
    ("Okay Jarvis remind me", (True, "remind me")),
    ("Jervis, read my mail", (True, "read my mail")),
    ("Jarvis.", (True, "")),
    ("Hi Jarvis summarise my inbox", (True, "summarise my inbox")),
    # Talking *about* Jarvis is not addressing it.
    ("Jarvis is the name of the assistant we are building", (False,
     "Jarvis is the name of the assistant we are building")),
    ("Hey Charlie, how is it going?", (False, "Hey Charlie, how is it going?")),
    ("Hey Travis, what's up", (False, "Hey Travis, what's up")),
    # "Jarvis" mentioned mid-sentence is not a wake phrase.
    ("I was telling my friend about the Jarvis project yesterday", (False,
     "I was telling my friend about the Jarvis project yesterday")),
])
def test_split_wake(text, expected):
    assert split_wake(text) == expected


# ---- detector + listener with a scripted scorer --------------------------------


class ScriptedScorer:
    """Fires once the given number of 80 ms chunks have been scored."""

    def __init__(self, fire_at_chunk):
        self.n, self.fire_at = 0, fire_at_chunk
        self.resets = 0

    def __call__(self, pcm):
        assert pcm.dtype == np.int16 and len(pcm) == 1280
        self.n += 1
        return 0.97 if self.n == self.fire_at else 0.01

    def reset(self):
        self.resets += 1


def test_listener_returns_wake_phrase_plus_request():
    # 1 s room noise, "hey jarvis" (0.8 s), request (2 s), silence.
    audio = np.concatenate([noise(1.0), voice(0.8), voice(2.0, 0.25), noise(3.0)])
    listener = WakeListener(WakeDetector(ScriptedScorer(fire_at_chunk=23), 0.5))  # fires at ~1.8 s
    woke = []
    out = listener.wait_for_request(reader(audio), on_wake=lambda: woke.append(1))
    assert woke == [1]
    secs = len(out) / SAMPLE_RATE
    assert 2.6 < secs < 4.6  # the wake phrase from the ring buffer + the whole request


def test_listener_ignores_audio_while_jarvis_is_talking():
    scorer = ScriptedScorer(fire_at_chunk=5)
    listener = WakeListener(WakeDetector(scorer, 0.5))
    with pytest.raises(EndOfAudio):  # never wakes: everything arrived while "talking"
        listener.wait_for_request(reader(voice(3)), paused=lambda: True)
    assert scorer.n == 0


def test_wake_with_nothing_after_returns_just_the_wake_phrase():
    audio = np.concatenate([noise(1.0), voice(0.8), noise(8.0)])
    listener = WakeListener(WakeDetector(ScriptedScorer(fire_at_chunk=22), 0.5),
                            EndpointSettings(no_speech_timeout_s=2))
    out = listener.wait_for_request(reader(audio))
    assert 1.0 < len(out) / SAMPLE_RATE <= 2.1  # only the ring buffer (≤ 2 s)


# ---- the real openWakeWord "hey jarvis" model -----------------------------------


def synth(text, voice_name="en-gb", speed=150):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import tempfile

    path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["espeak-ng", "-v", voice_name, "-s", str(speed), "-w", path, text], check=True)
    sr, a = wavfile.read(path)
    return (resample_poly(a.astype(np.float32), SAMPLE_RATE, sr) / 32768).astype(np.float32)


real_model = pytest.mark.skipif(
    shutil.which("espeak-ng") is None or pytest.importorskip("openwakeword") is None,
    reason="needs espeak-ng and openwakeword",
)


@real_model
def test_real_model_wakes_on_hey_jarvis_and_keeps_the_request():
    from jarvis.voice.wake import OpenWakeWordScorer

    request = synth("hey jarvis, what is the rent for unit four b")
    audio = np.concatenate([noise(1.5), request, noise(3.0)])
    listener = WakeListener(WakeDetector(OpenWakeWordScorer(), 0.5))
    out = listener.wait_for_request(reader(audio))
    # Wake phrase from the ring buffer plus the rest of the sentence after it.
    assert len(out) / SAMPLE_RATE > 0.8 * len(request) / SAMPLE_RATE


@real_model
def test_real_model_stays_asleep_for_ordinary_speech():
    from jarvis.voice.wake import OpenWakeWordScorer

    talk = np.concatenate([synth(t) for t in (
        "the rent is due on the first of the month",
        "please send the owner statement to doug",
        "we recorded the security deposit in buildium",
    )])
    listener = WakeListener(WakeDetector(OpenWakeWordScorer(), 0.5))
    with pytest.raises(EndOfAudio):
        listener.wait_for_request(reader(np.concatenate([noise(1), talk, noise(1)])))


# ---- the always-listening session ----------------------------------------------


class FakeStream:
    def __init__(self, audio):
        self.read = reader(audio)
        self.drained = 0

    def drain(self):
        self.drained += 1


class FakeWakeMic:
    def __init__(self, audio):
        self.stream = FakeStream(audio)

    @contextmanager
    def open(self):
        yield self.stream


class ScriptedListener:
    settings = EndpointSettings(no_speech_timeout_s=2)

    def __init__(self, clips):
        self.clips = list(clips)

    def noise_threshold(self):
        return 0.02

    def wait_for_request(self, read, paused, on_wake):
        import time
        while paused():  # like the real listener: nothing happens while Jarvis talks
            time.sleep(0.01)
        if not self.clips:
            raise KeyboardInterrupt  # end of test: like pressing Ctrl+C while quiet
        on_wake()
        return self.clips.pop(0)


def wake_session(env, heard, mic_audio=None):
    from jarvis.agent import Agent
    from jarvis.config import load_config
    from jarvis.llm import OllamaClient
    from jarvis.memory import Memory

    cfg = load_config()
    agent = Agent(cfg, OllamaClient(cfg.ollama_host), Memory(cfg.db_path))
    sp, log = fake_speaker(0.01)
    lines = []
    mic = FakeWakeMic(mic_audio if mic_audio is not None else noise(1))
    s = VoiceSession(agent, mic, FakeWhisper(heard), sp, printer_factory=NullPrinter,
                     on_tool=lambda n, a: None, confirm=lambda n, a: False,
                     out=lines.append, sound=lambda: None)
    return s, log, lines, mic


def test_wake_session_answers_and_ignores_false_alarms(jarvis_env):
    jarvis_env.queue_text("Unit 4B pays $2,150.")
    s, log, lines, mic = wake_session(jarvis_env, ["Hey Charlie, how's it going?",
                                                   "Hey Jarvis, what's the rent for 4B?"])
    assert s.run_wake(ScriptedListener([voice(1), voice(2)])) == 0
    s.speaker.wait(2)
    assert any("ignored a false alarm" in l for l in lines)
    assert [r["messages"][-1]["content"] for r in jarvis_env.requests] == ["what's the rent for 4B?"]
    assert ("done", "Unit 4B pays $2,150.") in log
    assert mic.stream.drained >= 1  # stale audio from while it was thinking is dropped


def test_wake_alone_asks_yes_then_hears_request(jarvis_env):
    jarvis_env.queue_text("You have two new emails.")
    follow_up = np.concatenate([noise(0.3), voice(1.5), noise(2.0)])
    s, log, lines, _ = wake_session(jarvis_env, ["Hey Jarvis.", "Any new emails?"], mic_audio=follow_up)
    assert s.run_wake(ScriptedListener([voice(1)])) == 0
    s.speaker.wait(2)
    assert ("done", "Yes?") in log
    assert jarvis_env.requests[0]["messages"][-1]["content"] == "Any new emails?"


def test_goodbye_ends_wake_mode(jarvis_env):
    s, log, lines, _ = wake_session(jarvis_env, ["Hey Jarvis, goodbye."])
    assert s.run_wake(ScriptedListener([voice(1), voice(1)])) == 0
    assert jarvis_env.requests == [] and ("done", "Goodbye.") in log
