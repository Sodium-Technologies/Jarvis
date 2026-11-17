"""A fake Ollama server that speaks the real /api/chat streaming protocol.

Tests queue scripted replies; the server records every request it receives so
tests can assert on exactly what Jarvis sent to the model.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeOllama:
    def __init__(self):
        self.replies: list[list[dict]] = []  # each reply is a list of stream chunks
        self.requests: list[dict] = []
        self.models = ["qwen3:8b"]
        self.reject_think = False

    def queue_text(self, text: str, thinking: str = "") -> None:
        chunks = []
        if thinking:
            chunks.append({"message": {"role": "assistant", "content": "", "thinking": thinking}})
        # Split into a few tokens to exercise streaming.
        for i in range(0, len(text), 4):
            chunks.append({"message": {"role": "assistant", "content": text[i : i + 4]}})
        chunks.append({"message": {"role": "assistant", "content": ""}, "done": True})
        self.replies.append(chunks)

    def queue_tool_call(self, name: str, arguments: dict) -> None:
        self.replies.append(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
                    }
                },
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
        )


def _handler(fake: FakeOllama):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/version":
                return self._json(200, {"version": "0.0.0-fake"})
            if self.path == "/api/tags":
                return self._json(200, {"models": [{"name": m} for m in fake.models]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                return self._json(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append(body)
            if body["model"] not in fake.models:
                return self._json(404, {"error": f"model '{body['model']}' not found"})
            if fake.reject_think and body.get("think"):
                return self._json(400, {"error": f"\"{body['model']}\" does not support thinking"})
            if not fake.replies:
                return self._json(500, {"error": "fake server: no reply queued"})
            chunks = fake.replies.pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for c in chunks:
                self.wfile.write((json.dumps(c) + "\n").encode())
                self.wfile.flush()

    return Handler


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()


@pytest.fixture
def jarvis_env(tmp_path, monkeypatch, fake_ollama):
    """Isolated Jarvis home pointing at the fake server."""
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("JARVIS_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("OLLAMA_HOST", fake_ollama.url)
    monkeypatch.delenv("JARVIS_MODEL", raising=False)
    return fake_ollama
