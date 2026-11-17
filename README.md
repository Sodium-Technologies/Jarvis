# Jarvis

A private AI assistant that runs on **your own computer**, using a local model
served by [Ollama](https://ollama.com). Your conversations, notes and (later)
emails and files are processed locally and never sent to a cloud AI.

> Status: chat, reasoning, memory, file search, Gmail (read-only) and **voice**.
> See [ROADMAP.md](ROADMAP.md) for what comes next.

## 1. Install Ollama and a model

1. Download and install Ollama from https://ollama.com/download (Windows, macOS or Linux).
2. Open a terminal (on Windows, use PowerShell) and pull the default model:

   ```
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```

   `nomic-embed-text` (~270 MB) is a small model used only to search your files by meaning.

   `qwen3:8b` (~5 GB) can reason step by step and call tools. It runs well with
   16 GB of RAM, and faster with a GPU that has 8 GB or more of memory. On a
   weaker machine use `qwen3:4b`. On a stronger one try `qwen3:14b` or `gpt-oss:20b`.

### Apple Silicon Macs (M1–M4)

Ollama uses the Mac's GPU (Metal) automatically. By default macOS lets the GPU
use only about two-thirds of the memory. On a 24 GB Mac that is roughly 16 GB,
and the model plus its working memory has to fit in it:

| Mac memory | Recommended model | Download | Notes |
|---|---|---|---|
| 16 GB | `qwen3:8b` | ~5 GB | |
| 24 GB | `qwen3:14b` | ~9 GB | Best balance; set `context_window = 16384` |
| 36 GB+ | `qwen3:30b-a3b` or `gpt-oss:20b` | 14–19 GB | |

On 24 GB, avoid `qwen3:30b-a3b` (~19 GB). It does not fit in GPU memory, so
it spills onto the processor and becomes slow.

## 2. Install Jarvis

You need Python 3.11 or newer. macOS's built-in Python is too old. Install a
newer one with [Homebrew](https://brew.sh): `brew install python@3.12 git`.
After that, use `python3.12` in place of `python` in the commands below.

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

## 5. Let Jarvis find things in your files

Choose the folders Jarvis may search. It reads only these folders.

```
jarvis index add ~/Documents
jarvis index add ~/Library/CloudStorage/OneDrive-Personal
jarvis index add "~/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive"
jarvis index list      # folders and counts
jarvis index update    # pick up new/changed/deleted files (fast; only changes are re-read)
jarvis search "security deposit Maria Lopez"   # search directly, without the AI
```

On a Mac, OneDrive and Google Drive sync into `~/Library/CloudStorage/`. Run
`ls ~/Library/CloudStorage` to see the exact folder names. The first time,
macOS will ask whether Terminal may access Documents or files managed by
OneDrive or Google Drive. Click **Allow**.

In chat, just ask: *"Find the lease for unit 4B and tell me the rent"*, or
*"Where is the October occupancy report?"* Jarvis searches, opens the file, and
names the file it used.

**What can be read:** PDF (with a text layer), Word `.docx`, Excel `.xlsx`,
CSV, text, Markdown, HTML, and saved emails `.eml`. Every other file can still
be found by its **name**.

**Known limits (for now):**
- Scanned PDFs (images with no text) are found by name only. Text recognition (OCR) is planned.
- Old `.doc`/`.xls` formats and Outlook `.msg` files are found by name only.
- Native Google Docs and Sheets are only shortcut files on disk (`.gdoc`, `.gsheet`).
  They are found by name only, until the Google Drive connection in a later step.
- **Online-only** cloud files are found by name only, so that indexing doesn't
  download your whole drive. To include their content, right-click the folder in
  Finder, choose *Keep Downloaded* or *Always Keep on This Device*, then run
  `jarvis index update`.

`jarvis index remove <folder>` deletes everything Jarvis extracted from that folder.

## 6. Connect Gmail (read-only)

Jarvis signs in to Gmail with an **app password**. That is a separate
16-letter password just for Jarvis, which you can revoke at any time without
changing your real password.

1. Turn on 2-Step Verification for the Google account: https://myaccount.google.com/signinoptions/twosv
2. Create an app password named "Jarvis": https://myaccount.google.com/apppasswords
3. Run the guided setup. It checks the login before saving anything:

```
jarvis mail add          # asks for address, a short name, your name/role/client, app password
jarvis mail list
jarvis mail test bsf
jarvis mail search bsf "from:doug has:attachment newer_than:30d"
jarvis mail remove bsf   # disconnects and deletes the saved password
```

Each account records **who you are on it** (for example "Naveed Ahmed Khan,
Bookkeeper for Black Squirrel Flats"). Jarvis keeps each client's mail and
identity separate.

In chat, ask things like:
- *"Any emails from Doug this week about deposits?"*
- *"Find the Elm Plumbing invoice and tell me the amount on the PDF."*
- *"Summarise the thread about unit 2's security deposit."*

Searches use Gmail's own syntax (`from:`, `subject:`, `has:attachment`,
`filename:pdf`, `newer_than:7d`, `after:2025/10/01`), exactly like the Gmail search box.

**How the safety works**
- **Read-only in code.** Jarvis opens the mailbox in IMAP read-only mode and
  fetches with `PEEK`, so nothing is marked read, moved or deleted. There are no
  send or delete tools yet. Note that Google itself grants an app password full
  mailbox access, so the read-only limit is enforced by Jarvis's code. Keep the
  app password secret, and revoke it if your Mac is lost.
- **Password storage.** The app password is stored in the **macOS Keychain**, never
  in Jarvis's files.
- **Email is data, not instructions.** Jarvis is told never to follow
  instructions written inside emails or attachments.
- **Where mail goes.** Mail is read live from Gmail and processed only by your
  local model. Replies that quote emails are saved in your local chat history
  (`~/.jarvis/jarvis.db`), so keep FileVault on.

Outlook / Microsoft 365 (LUXE) is coming in Step 3b.

## 7. Talk to Jarvis (voice)

Free and fully offline. **Whisper** (open-source speech recognition) turns your
voice into text on your Mac, and the built-in macOS voices read the replies aloud.
No audio leaves the computer.

```
pip install -e ".[dev,voice]"   # one time: speech recognition + microphone support
jarvis voice --test              # checks the microphone, speech recognition and voice
jarvis voice                     # press Enter, speak, pause; Jarvis answers out loud
jarvis voice --hands-free        # keeps listening after each reply
```

- The first run downloads the Whisper model once (~500 MB for `small.en`).
- macOS will ask to let Terminal use the **microphone**. Click Allow. If you missed
  it: System Settings → Privacy & Security → Microphone → Terminal.
- You can still type instead of speaking. Press **Enter while Jarvis is talking**
  to interrupt it. Say **"goodbye"** or type `/exit` to quit.
- Approvals for risky actions are always **typed**. A misheard "yes" never
  approves anything.
- In voice mode Jarvis skips step-by-step reasoning by default, because it adds
  10–20 seconds of silence. Use `jarvis voice --think` for hard questions.

**A better voice (free):** System Settings → Accessibility → Spoken Content →
System voice → Manage Voices. Download e.g. *Daniel (Enhanced)* or *Jamie
(Premium)* under English (UK). Then set `tts_voice` in `config.toml` to the exact
name shown by `say -v '?'`.

**Speaking Urdu or mixing languages:** set `stt_model = "small"` and `stt_language = ""`.

## Configuration

Optional. Create `~/.jarvis/config.toml` (on Windows, `C:\Users\<you>\.jarvis\config.toml`):

```toml
model = "qwen3:8b"
think = true            # let reasoning models think before answering
show_thinking = false   # print the reasoning
temperature = 0.3
context_window = 8192   # raise if you have spare RAM/VRAM
embed_model = "nomic-embed-text"   # "" turns off search by meaning
min_similarity = 0.45   # lower if search misses files, raise if it returns unrelated ones
max_file_mb = 50        # bigger files are indexed by name only
stt_model = "small.en"  # speech recognition: tiny.en, base.en, small.en, medium.en
tts_voice = "Daniel"    # any voice from: say -v '?'
tts_rate = 0            # words per minute, 0 = normal
voice_silence_ms = 900  # how long a pause means "I'm done talking"
```

Environment variables override the file: `JARVIS_MODEL`, `OLLAMA_HOST`,
`JARVIS_HOME` (data folder), `JARVIS_CONFIG` (config file path).

All data lives in `~/.jarvis/`: `jarvis.db` (conversations, notes) and
`index.db` (text extracted from your files). Both are plain SQLite files that
never leave your computer.

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

The tests run against a fake Ollama server (including fake embeddings) and a
fake Gmail IMAP server (`tests/fake_imap.py`), (`tests/conftest.py`) that speaks
the same streaming protocol, so they need no GPU and no model download.

Code layout:

| File | Purpose |
|---|---|
| `jarvis/config.py` | Settings (file + env vars) |
| `jarvis/llm.py` | Ollama client: streaming, reasoning, tool calls |
| `jarvis/agent.py` | Agent loop: model → tools → model → answer |
| `jarvis/tools.py` | Tool registry, confirmation gate, built-in tools |
| `jarvis/memory.py` | SQLite: conversations and learned notes |
| `jarvis/extract.py` | Text from PDF, Word, Excel, CSV, HTML, .eml |
| `jarvis/index.py` | File index: scanning, chunks, keyword + meaning search |
| `jarvis/mail/gmail.py` | Read-only Gmail over IMAP (X-GM-RAW search, threads, attachments) |
| `jarvis/mail/accounts.py` | Account profiles + keychain-stored passwords |
| `jarvis/mail/tools.py` | Email tools for the agent |
| `jarvis/voice/listen.py` | Microphone + end-of-speech detection |
| `jarvis/voice/stt.py` | Whisper speech-to-text |
| `jarvis/voice/speak.py` | Speakable text, sentence streaming, macOS `say` |
| `jarvis/voice/session.py` | The talk loop |
| `jarvis/cli.py` | `jarvis` command |
