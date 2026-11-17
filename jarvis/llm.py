"""Thin client for a locally running Ollama server.

Only Ollama's documented HTTP API is used (/api/version, /api/tags, /api/chat),
so nothing leaves the machine unless OLLAMA_HOST points elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterator

import httpx


class LLMError(RuntimeError):
    """Raised for any problem talking to the local model server."""


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.thinking:
            msg["thinking"] = self.thinking
        if self.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        return msg


# Callback receives (kind, text) where kind is "content" or "thinking".
StreamCallback = Callable[[str, str], None]


class OllamaClient:
    def __init__(self, host: str, timeout: float = 300.0):
        self.host = host.rstrip("/")
        self._http = httpx.Client(base_url=self.host, timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self) -> None:
        self._http.close()

    # ---- health ----------------------------------------------------------

    def version(self) -> str:
        return self._get("/api/version").get("version", "unknown")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def has_model(self, model: str) -> bool:
        names = self.list_models()
        # "qwen3" matches "qwen3:latest"
        return model in names or f"{model}:latest" in names

    # ---- embeddings ------------------------------------------------------

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        try:
            resp = self._http.post("/api/embed", json={"model": model, "input": texts})
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is it running? (try: ollama serve)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The embedding model took too long to respond (timeout).") from exc
        if resp.status_code != 200:
            raise LLMError(self._error_text(resp))
        vectors = resp.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise LLMError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    # ---- chat ------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        think: bool = False,
        options: dict | None = None,
        on_token: StreamCallback | None = None,
    ) -> ChatResult:
        """Run one chat turn. Streams tokens to on_token if given."""
        payload: dict = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
        if think:
            payload["think"] = True
        if options:
            payload["options"] = options

        try:
            return self._stream_chat(payload, on_token)
        except LLMError as exc:
            # Models without reasoning support reject "think"; retry once without it.
            if think and "does not support thinking" in str(exc):
                payload.pop("think")
                return self._stream_chat(payload, on_token)
            raise

    def _stream_chat(self, payload: dict, on_token: StreamCallback | None) -> ChatResult:
        result = ChatResult()
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise LLMError(self._error_text(resp))
                for chunk in self._iter_ndjson(resp.iter_lines()):
                    if "error" in chunk:
                        raise LLMError(chunk["error"])
                    msg = chunk.get("message") or {}
                    if msg.get("thinking"):
                        result.thinking += msg["thinking"]
                        if on_token:
                            on_token("thinking", msg["thinking"])
                    if msg.get("content"):
                        result.content += msg["content"]
                        if on_token:
                            on_token("content", msg["content"])
                    for call in msg.get("tool_calls") or []:
                        fn = call.get("function", {})
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):  # some models return a JSON string
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"_raw": args}
                        result.tool_calls.append(ToolCall(fn.get("name", ""), args))
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is it running? (try: ollama serve)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The model took too long to respond (timeout).") from exc
        return result

    # ---- helpers ---------------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            resp = self._http.get(path)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is it running? (try: ollama serve)"
            ) from exc
        if resp.status_code != 200:
            raise LLMError(self._error_text(resp))
        return resp.json()

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        try:
            return resp.json().get("error", resp.text)
        except ValueError:
            return f"HTTP {resp.status_code}: {resp.text[:200]}"

    @staticmethod
    def _iter_ndjson(lines: Iterator[str]) -> Iterator[dict]:
        for line in lines:
            line = line.strip()
            if line:
                yield json.loads(line)
