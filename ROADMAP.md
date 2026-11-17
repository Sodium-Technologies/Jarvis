# Roadmap

Each step ships working and tested before the next starts.

## Step 1: Foundation ✅
- Local model via Ollama (default `qwen3:8b`), streaming replies
- Reasoning ("thinking") support with a fallback for models that cannot think
- Tool calling with a **confirmation gate** for anything that changes the outside world
- SQLite memory: conversations persist; `remember` / `forget` notes ("learning")
- CLI: `doctor`, `chat`, `ask`, `history`, `notes`
- Test suite against a fake Ollama server

## Step 2: Find things in files and drives (local search)
- Index chosen folders: local disk, plus OneDrive / Google Drive sync folders
- Read PDF, Word, Excel, CSV, text and email exports (.eml/.msg)
- Embeddings from a local model (`nomic-embed-text` via Ollama) plus keyword search
- Tools: `search_files`, `read_file`. Answers cite the file path.

## Step 3: Email (read-only first)
- **Client profiles**, so each client's identity, mailbox and tools stay separate:
  - LUXE MANAGEMENT LLC: Outlook / Microsoft 365 via Microsoft Graph, acting as Yasir Ali
  - Black Squirrel Flats: Gmail API, acting as Naveed Ahmed Khan
- Tools: search mail, read a thread, summarise the inbox, find attachments
- OAuth tokens stored in the OS keychain, never in the repo or the database

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
