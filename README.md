# Jarvis

A private AI assistant that runs on **your own computer**, using a local model
served by [Ollama](https://ollama.com). Your conversations, notes and (later)
emails and files are processed locally and never sent to a cloud AI.

> Status: **Step 1 – foundation** (chat, reasoning, tools, memory).
> See [ROADMAP.md](ROADMAP.md) for what comes next.

## 1. Install Ollama and a model

1. Download and install Ollama from https://ollama.com/download (Windows, macOS or Linux).
2. Open a terminal (on Windows, use PowerShell) and pull the default model:

   ```
   ollama pull qwen3:8b
   ```

   `qwen3:8b` (~5 GB) can reason step by step and call tools. It runs well with
   16 GB of RAM, and faster with a GPU that has 8 GB or more of memory. On a
   weaker machine use `qwen3:4b`. On a stronger one try `qwen3:14b` or `gpt-oss:20b`.

## 2. Install Jarvis

You need Python 3.11 or newer.

```
git clone https://github.com/sodium-technologies/jarvis.git
cd jarvis
python -m venv .venv
```

Activate the environment. On Windows: `.venv\Scripts\activate`. On macOS or Linux: `source .venv/bin/activate`. Then:

```
pip install -e ".[dev]"
```

## 3. Check everything works

```
jarvis doctor
```

You should see three `OK` lines: Ollama is running, the model is installed, and a test reply.

## 4. Use it

```
jarvis chat              # interactive chat
jarvis chat --resume     # continue the last conversation
jarvis ask "Summarise the difference between a 1099 and a W-2"
jarvis history           # past conversations
jarvis notes             # facts Jarvis has learned about you
jarvis --model qwen3:4b chat   # use a different model for one run
```

Inside chat: `/think` shows or hides the model's reasoning, `/exit` quits.

**Learning:** Jarvis keeps notes about you. Tell it something like *"Remember
that Black Squirrel Flats uses Buildium"* and it saves a note. Saved notes go
into every future conversation. Deleting a note always asks for your
confirmation first.

## Configuration

Optional. Create `~/.jarvis/config.toml` (on Windows, `C:\Users\<you>\.jarvis\config.toml`):

```toml
model = "qwen3:8b"
think = true            # let reasoning models think before answering
show_thinking = false   # print the reasoning
temperature = 0.3
context_window = 8192   # raise if you have spare RAM/VRAM
```

Environment variables override the file: `JARVIS_MODEL`, `OLLAMA_HOST`,
`JARVIS_HOME` (data folder), `JARVIS_CONFIG` (config file path).

All data lives in `~/.jarvis/jarvis.db` (SQLite). Back it up or delete it as you like.

## Safety design

- **Local only.** Jarvis talks only to Ollama at `OLLAMA_HOST` (default `localhost`).
- **Confirmation gate.** Any tool that changes something outside Jarvis must be
  registered with `confirm=True`. That covers sending email, deleting, and
  anything financial. Jarvis then shows you the exact action and waits for your "y"
  before running it. This is deliberate: it keeps a human approval step on anything
  sent to tenants, owners or clients.

## Development

```
pytest
```

The tests run against a fake Ollama server (`tests/conftest.py`) that speaks
the same streaming protocol, so they need no GPU and no model download.

Code layout:

| File | Purpose |
|---|---|
| `jarvis/config.py` | Settings (file + env vars) |
| `jarvis/llm.py` | Ollama client: streaming, reasoning, tool calls |
| `jarvis/agent.py` | Agent loop: model → tools → model → answer |
| `jarvis/tools.py` | Tool registry, confirmation gate, built-in tools |
| `jarvis/memory.py` | SQLite: conversations and learned notes |
| `jarvis/cli.py` | `jarvis` command |
