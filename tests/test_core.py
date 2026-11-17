from pathlib import Path

import pytest

from jarvis.agent import Agent
from jarvis.cli import main
from jarvis.config import load_config
from jarvis.llm import LLMError, OllamaClient
from jarvis.memory import Memory
from jarvis.tools import Tool, ToolRegistry


def make_agent(env, **kw):
    cfg = load_config()
    llm = OllamaClient(cfg.ollama_host)
    mem = Memory(cfg.db_path)
    return Agent(cfg, llm, mem, **kw), mem


# ---- config ---------------------------------------------------------------


def test_config_file_and_env_precedence(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('model = "llama3.2"\nthink = false\ntemperature = 0.9\n')
    monkeypatch.setenv("JARVIS_CONFIG", str(f))
    monkeypatch.setenv("JARVIS_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    cfg = load_config()
    assert cfg.model == "gpt-oss:20b"  # env beats file
    assert cfg.think is False and cfg.temperature == 0.9
    assert cfg.ollama_host == "http://0.0.0.0:11434"


def test_config_rejects_unknown_keys(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('modle = "typo"\n')
    monkeypatch.setenv("JARVIS_CONFIG", str(f))
    with pytest.raises(ValueError, match="modle"):
        load_config()


# ---- llm client -----------------------------------------------------------


def test_streaming_reply_and_thinking(jarvis_env):
    jarvis_env.queue_text("Hello there, Naveed.", thinking="User greeted me.")
    seen = []
    r = OllamaClient(jarvis_env.url).chat(
        "qwen3:8b", [{"role": "user", "content": "hi"}], think=True,
        on_token=lambda k, t: seen.append(k),
    )
    assert r.content == "Hello there, Naveed."
    assert r.thinking == "User greeted me."
    assert seen[0] == "thinking" and seen.count("content") > 1
    assert jarvis_env.requests[0]["think"] is True and jarvis_env.requests[0]["stream"] is True


def test_falls_back_when_model_cannot_think(jarvis_env):
    jarvis_env.reject_think = True
    jarvis_env.queue_text("ok")
    r = OllamaClient(jarvis_env.url).chat("qwen3:8b", [], think=True)
    assert r.content == "ok"
    assert "think" not in jarvis_env.requests[-1]


def test_clear_error_when_ollama_down():
    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        OllamaClient("http://127.0.0.1:9").version()


def test_missing_model_error(jarvis_env):
    with pytest.raises(LLMError, match="not found"):
        OllamaClient(jarvis_env.url).chat("nope", [])


# ---- agent + tools --------------------------------------------------------


def test_agent_tool_loop_and_persistence(jarvis_env):
    jarvis_env.queue_tool_call("get_current_datetime", {})
    jarvis_env.queue_text("It is Tuesday.")
    agent, mem = make_agent(jarvis_env)
    result = agent.ask("What day is it?")
    assert result.content == "It is Tuesday."

    # Second request must contain the tool result fed back to the model.
    second = jarvis_env.requests[1]["messages"]
    assert second[0]["role"] == "system"
    assert second[-1]["role"] == "tool" and second[-1]["tool_name"] == "get_current_datetime"
    assert "tools" in jarvis_env.requests[0]

    # Conversation survives a restart.
    roles = [m["role"] for m in mem.get_messages(agent.conversation_id)]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_remember_note_reaches_future_conversations(jarvis_env):
    jarvis_env.queue_tool_call("remember", {"fact": "Works for Black Squirrel Flats as bookkeeper."})
    jarvis_env.queue_text("Noted.")
    agent, mem = make_agent(jarvis_env)
    agent.ask("Remember that I am the bookkeeper for Black Squirrel Flats.")

    jarvis_env.queue_text("You are a bookkeeper.")
    fresh = Agent(agent.config, agent.llm, mem)  # brand-new conversation
    fresh.ask("What do I do?")
    system = jarvis_env.requests[-1]["messages"][0]["content"]
    assert "Black Squirrel Flats" in system


def test_confirmation_required_tool_is_blocked_when_declined(jarvis_env):
    calls = []
    reg = ToolRegistry()
    reg.register(Tool("send_email", "Send", {"type": "object", "properties": {}},
                      lambda: calls.append(1) or "sent", confirm=True))
    jarvis_env.queue_tool_call("send_email", {})
    jarvis_env.queue_text("Okay, I did not send it.")
    agent, _ = make_agent(jarvis_env, tools=reg)
    agent.ask("send it", confirm=lambda n, a: False)
    assert calls == []
    assert "declined" in jarvis_env.requests[1]["messages"][-1]["content"]

    jarvis_env.queue_tool_call("send_email", {})
    jarvis_env.queue_text("Sent.")
    agent.ask("send it", confirm=lambda n, a: True)
    assert calls == [1]


def test_tool_errors_are_reported_not_raised(jarvis_env):
    jarvis_env.queue_tool_call("forget", {"wrong_arg": 1})
    jarvis_env.queue_tool_call("no_such_tool", {})
    jarvis_env.queue_text("Sorry.")
    agent, _ = make_agent(jarvis_env)
    agent.ask("x", confirm=lambda n, a: True)
    tool_msgs = [m for m in jarvis_env.requests[-1]["messages"] if m["role"] == "tool"]
    assert "bad arguments" in tool_msgs[0]["content"]
    assert "no tool named" in tool_msgs[1]["content"]


def test_runaway_tool_loop_is_capped(jarvis_env):
    agent, _ = make_agent(jarvis_env)
    for _ in range(agent.config.max_tool_steps):
        jarvis_env.queue_tool_call("get_current_datetime", {})
    r = agent.ask("loop")
    assert "too many tool steps" in r.content


# ---- CLI ------------------------------------------------------------------


def test_cli_doctor_and_ask(jarvis_env, capsys):
    jarvis_env.queue_text("ready")
    assert main(["doctor"]) == 0
    assert "Test reply: 'ready'" in capsys.readouterr().out

    jarvis_env.queue_text("Four.")
    assert main(["ask", "What", "is", "2+2?"]) == 0
    assert "Four." in capsys.readouterr().out

    assert main(["history"]) == 0
    assert "What is 2+2?" in capsys.readouterr().out


def test_cli_doctor_reports_missing_model(jarvis_env, capsys):
    jarvis_env.models = ["llama3.2:latest"]
    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "ollama pull qwen3:8b" in out and "llama3.2:latest" in out
