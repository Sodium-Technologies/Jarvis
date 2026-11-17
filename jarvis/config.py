"""Configuration loading.

Precedence (highest first): environment variables, config file, defaults.

Config file location: $JARVIS_CONFIG, else ~/.jarvis/config.toml.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = """\
You are Jarvis, a private assistant running entirely on the user's own computer.
Be accurate and concise. If you are not sure, say so instead of guessing.
Use the tools you are given when they help; never invent tool results.
Never take an action that sends, deletes, or pays for anything without the
user's explicit confirmation."""


def _default_home() -> Path:
    return Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))


@dataclass
class Config:
    ollama_host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    # Ask reasoning-capable models (qwen3, deepseek-r1, gpt-oss...) to think first.
    think: bool = True
    # Show the model's reasoning in the terminal.
    show_thinking: bool = False
    temperature: float = 0.3
    context_window: int = 8192
    request_timeout: float = 300.0
    max_tool_steps: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    home: Path = field(default_factory=_default_home)

    @property
    def db_path(self) -> Path:
        return self.home / "jarvis.db"


_ENV_MAP = {
    "OLLAMA_HOST": "ollama_host",
    "JARVIS_MODEL": "model",
    "JARVIS_THINK": "think",
    "JARVIS_SHOW_THINKING": "show_thinking",
}


def _coerce(name: str, value):
    typ = {f.name: f.type for f in fields(Config)}[name]
    if typ in ("bool", bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if typ in ("int", int):
        return int(value)
    if typ in ("float", float):
        return float(value)
    if typ in ("Path", Path):
        return Path(value).expanduser()
    return value


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or Path(os.environ.get("JARVIS_CONFIG", cfg.home / "config.toml"))
    valid = {f.name for f in fields(Config)}

    if path.exists():
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {', '.join(sorted(unknown))}")
        for key, value in data.items():
            setattr(cfg, key, _coerce(key, value))

    for env, key in _ENV_MAP.items():
        if env in os.environ:
            setattr(cfg, key, _coerce(key, os.environ[env]))

    # Ollama accepts OLLAMA_HOST without a scheme (e.g. "0.0.0.0:11434").
    if not cfg.ollama_host.startswith(("http://", "https://")):
        cfg.ollama_host = "http://" + cfg.ollama_host
    cfg.ollama_host = cfg.ollama_host.rstrip("/")
    return cfg
