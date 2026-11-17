# Roadmap

Each step ships working and tested before the next starts.

## Step 1: Foundation ✅
- Local model via Ollama (default `qwen3:8b`), streaming replies
- Reasoning ("thinking") support with a fallback for models that cannot think
- Tool calling with a **confirmation gate** for anything that changes the outside world
- SQLite memory: conversations persist; `remember` / `forget` notes ("learning")
- CLI: `doctor`, `chat`, `ask`, `history`, `notes`
- Test suite against a fake Ollama server

## Step 2: Find things in files and drives (local search) ✅
- Index chosen folders: local disk, plus OneDrive / Google Drive sync folders
- Read PDF, Word, Excel, CSV, text, HTML and saved emails (.eml)
- Hybrid search: SQLite full-text (BM25) + local embeddings (`nomic-embed-text`),
  merged by reciprocal rank fusion, with a similarity floor so nonsense finds nothing
- Incremental re-scans; online-only cloud files are not force-downloaded
- Tools: `search_documents`, `find_files`, `read_file` (restricted to indexed folders)
- CLI: `jarvis index add|remove|update|list`, `jarvis search`
- Later: OCR for scanned PDFs, `.msg`/`.doc`/`.xls`, automatic re-index in background

## Step 3: Gmail, read-only ✅
- Account profiles with per-account identity (name, role, client), so clients stay separate
- Gmail over IMAP with an app password (stored in the macOS Keychain). Chosen over
  the Gmail API because unverified OAuth apps must sign in again every 7 days on @gmail.com.
- Read-only enforced in code: EXAMINE + BODY.PEEK, no mutating IMAP commands; tested
- Tools: `search_email` (Gmail syntax), `read_email`, `read_email_thread`, `read_email_attachment`
- Email text is marked as untrusted third-party data in tool output and the system prompt
- CLI: `jarvis mail add|list|test|remove|search`

## Step 3b: Outlook / Microsoft 365, read-only ⏸ (postponed until Julie can approve the app)
- LUXE MANAGEMENT LLC mailbox via Microsoft Graph (device-code sign-in), acting as Yasir Ali
- May need Julie (as LUXE's Microsoft 365 admin) to approve the app for her organization
- Same four tools, so the agent does not care which provider an account uses

## Voice ✅ (added on request)
- `jarvis voice`: press Enter, speak, pause; replies spoken sentence by sentence as they are written
- Speech-to-text: Whisper via faster-whisper, local and free (`small.en` default; multilingual option)
- Text-to-speech: macOS `say` (default voice Daniel; free Enhanced/Premium voices supported)
- Silence detection that adapts to room noise; interrupt by pressing Enter; hands-free mode
- Confirmations stay typed; reasoning is never read aloud; markdown/code/paths cleaned for speech
- Later: "Hey Jarvis" wake word (openWakeWord, free), more natural neural voices (Kokoro/Piper)

## Step 4: Email replies
- Draft replies into the mailbox's **Drafts** folder (default behaviour)
- Sending only via a confirmed tool call, showing the exact recipients and text
- Per-client signature and tone

## Step 5: Coding helper
- Read, edit and create files inside an allowed workspace folder
- Run commands and tests, with confirmation

## Step 6: Better learning
- Automatically suggest facts worth remembering at the end of a chat
- Semantic search over past conversations ("what did we decide about X?")

## Later / optional
- Desktop or web UI
- Buildium API connector (needs API access from each client's Buildium account)
- Scheduled jobs (e.g. a morning inbox summary)

## Compliance notes to respect while building
- Client mailboxes and drives hold tenant and owner personal and financial data.
  Maryland's Personal Information Protection Act and D.C.'s data-breach law
  expect reasonable security for it. That is why Jarvis keeps processing local,
  stores data on your disk only, and should run on an encrypted disk
  (BitLocker or FileVault).
- Connect a client's mailbox or drive only with that client's authorization.
  Microsoft 365 may need Julie's admin consent for the app.
- Jarvis drafts. A human approves anything sent to tenants, owners or
  authorities: notices, ledgers, tax filings.
