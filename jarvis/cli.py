"""Command-line interface.

    jarvis doctor            check that Ollama and the model are ready
    jarvis chat              interactive chat (new conversation)
    jarvis chat --resume     continue the most recent conversation
    jarvis ask "question"    one-shot question
    jarvis history           list past conversations
    jarvis notes             show what Jarvis has learned
    jarvis index add PATH    add a folder to the document index
    jarvis index update      scan folders for new/changed files
    jarvis index list        show indexed folders and counts
    jarvis index remove PATH stop indexing a folder
    jarvis search "query"    search documents directly (no AI)
    jarvis mail add          connect a Gmail account (read-only)
    jarvis mail list         show connected accounts
    jarvis mail test NAME    check an account can log in
    jarvis mail remove NAME  disconnect an account and delete its saved password
    jarvis mail search NAME "gmail query"   search mail directly (no AI)
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from . import __version__
from .agent import Agent
from .config import load_config
from .index import FileIndex
from .llm import LLMError, OllamaClient
from .mail.accounts import AccountStore, MailAccount
from .mail.gmail import GmailIMAP, MailError
from .memory import Memory

DIM, CYAN, YELLOW, RED, GREEN, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[32m", "\033[0m"


class Printer:
    """Streams model output, showing reasoning dimmed when enabled."""

    def __init__(self, show_thinking: bool):
        self.show_thinking = show_thinking
        self.mode: str | None = None

    def __call__(self, kind: str, text: str) -> None:
        if kind == "thinking" and not self.show_thinking:
            if self.mode != "thinking":
                print(f"{DIM}(thinking...){RESET}", end="\r", flush=True)
                self.mode = "thinking"
            return
        if kind != self.mode:
            if self.mode == "thinking" and not self.show_thinking:
                print(" " * 16, end="\r")
            elif self.mode is not None:
                print()
            print(DIM if kind == "thinking" else RESET, end="")
            self.mode = kind
        print(text, end="", flush=True)

    def finish(self) -> None:
        print(RESET)
        self.mode = None


def on_tool(name: str, args: dict) -> None:
    print(f"\n{CYAN}[tool] {name}({json.dumps(args, ensure_ascii=False)}){RESET}")


def confirm(name: str, args: dict) -> bool:
    print(f"\n{YELLOW}Jarvis wants to run: {name}")
    print(json.dumps(args, indent=2, ensure_ascii=False))
    answer = input(f"Allow? [y/N] {RESET}").strip().lower()
    return answer in ("y", "yes")


def cmd_doctor(cfg, llm: OllamaClient) -> int:
    ok = True
    print(f"Config home : {cfg.home}")
    print(f"Ollama host : {cfg.ollama_host}")
    print(f"Model       : {cfg.model}")
    try:
        print(f"{GREEN}OK{RESET}   Ollama is running (version {llm.version()})")
    except LLMError as exc:
        print(f"{RED}FAIL{RESET} {exc}")
        return 1
    models = llm.list_models()
    if llm.has_model(cfg.model):
        print(f"{GREEN}OK{RESET}   Model '{cfg.model}' is installed")
    else:
        ok = False
        print(f"{RED}FAIL{RESET} Model '{cfg.model}' is not installed. Run: ollama pull {cfg.model}")
        if models:
            print(f"     Installed models: {', '.join(models)}")
    if cfg.embed_model:
        if llm.has_model(cfg.embed_model):
            print(f"{GREEN}OK{RESET}   Embedding model '{cfg.embed_model}' is installed")
        else:
            print(f"{YELLOW}WARN{RESET} Embedding model '{cfg.embed_model}' is missing; document search "
                  f"will use keywords only. Run: ollama pull {cfg.embed_model}")
    if ok:
        try:
            r = llm.chat(cfg.model, [{"role": "user", "content": "Reply with just the word: ready"}])
            print(f"{GREEN}OK{RESET}   Test reply: {r.content.strip()[:60]!r}")
        except LLMError as exc:
            ok = False
            print(f"{RED}FAIL{RESET} Test chat failed: {exc}")
    return 0 if ok else 1


def run_turn(agent: Agent, text: str, show_thinking: bool) -> None:
    printer = Printer(show_thinking)
    agent.ask(text, on_token=printer, on_tool=on_tool, confirm=confirm)
    printer.finish()


def open_index(cfg, llm) -> FileIndex:
    return FileIndex(cfg.index_path, llm, cfg.embed_model, cfg.max_file_mb, cfg.min_similarity)


def cmd_index(cfg, llm, args) -> int:
    index = open_index(cfg, llm)
    try:
        if args.action == "add":
            for folder in args.paths:
                try:
                    path = index.add_folder(folder)
                    print(f"Added {path}")
                except ValueError as exc:
                    print(f"{RED}{exc}{RESET}")
                    return 1
            print("Scanning…")
            return _print_update(index)
        if args.action == "remove":
            for folder in args.paths:
                print(("Removed " if index.remove_folder(folder) else "Not indexed: ") + folder)
            return 0
        if args.action == "update":
            if not index.folders():
                print("No folders yet. Add one with: jarvis index add ~/Documents")
                return 1
            return _print_update(index)
        # list
        folders = index.folders()
        if not folders:
            print("No folders indexed yet. Add one with: jarvis index add ~/Documents")
            return 0
        for f in folders:
            print(f"  {f}")
        st = index.stats()
        detail = ", ".join(f"{n} {k}" for k, n in sorted(st["by_status"].items()))
        print(f"{st['files']} files ({detail}); {st['chunks']} passages, {st['embedded']} searchable by meaning")
        return 0
    finally:
        index.close()


def _print_update(index: FileIndex) -> int:
    tty = sys.stdout.isatty()

    def progress(msg: str) -> None:
        # Overwrite one line in a terminal; plain lines when output is redirected.
        if tty:
            print(f"\r{DIM}  {msg}{RESET}\033[K", end="", flush=True)

    stats = index.update(progress=progress)
    if tty:
        print("\r\033[K", end="")
    print(f"{GREEN}Done.{RESET} {stats.scanned} files scanned: {stats.indexed} new/changed, "
          f"{stats.unchanged} unchanged, {stats.removed} removed, {stats.embedded} passages embedded.")
    if stats.errors:
        print(f"{YELLOW}{len(stats.errors)} file(s) could not be read:{RESET}")
        for path, err in stats.errors[:15]:
            print(f"  {path}: {err}")
        if len(stats.errors) > 15:
            print(f"  … and {len(stats.errors) - 15} more")
    if stats.embed_warning:
        print(f"{YELLOW}{stats.embed_warning}{RESET}")
    return 0


def cmd_search(cfg, llm, query: str) -> int:
    index = open_index(cfg, llm)
    try:
        hits = index.search(query, limit=10)
        if not hits:
            print("No matches.")
        for h in hits:
            print(f"{CYAN}{h.path}{RESET}  {DIM}{h.modified}{RESET}\n  {h.snippet}\n")
        return 0
    finally:
        index.close()


APP_PASSWORD_HELP = """\
Jarvis reads Gmail with an *app password* (a separate 16-letter password just
for Jarvis, which you can revoke at any time without changing your real one).
  1. Turn on 2-Step Verification: https://myaccount.google.com/signinoptions/twosv
  2. Create an app password:      https://myaccount.google.com/apppasswords
     (name it "Jarvis"), then paste the 16 letters below.
Jarvis opens the mailbox read-only: it cannot send, delete or mark mail as read."""


def _ask(prompt: str, default: str = "") -> str:
    value = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return value or default


def cmd_mail(cfg, args) -> int:
    store = AccountStore(cfg.db_path)
    try:
        if args.action == "list":
            accounts = store.list()
            if not accounts:
                print("No email accounts connected. Add one with: jarvis mail add")
            for a in accounts:
                print(f"  {a.describe()}")
            return 0

        if args.action == "add":
            print(APP_PASSWORD_HELP + "\n")
            address = _ask("Gmail address").lower()
            if "@" not in address:
                print(f"{RED}That doesn't look like an email address.{RESET}")
                return 1
            default_name = "personal" if not store.list() else ""
            account = MailAccount(
                name=_ask("Short name for this account (e.g. personal, bsf)", default_name).lower(),
                provider="gmail",
                address=address,
                owner_name=_ask("Your name as used on this account (e.g. Naveed Ahmed Khan)"),
                role=_ask("Your role for this account (e.g. Bookkeeper), optional"),
                organization=_ask("Client / organization (e.g. Black Squirrel Flats), optional"),
                imap_host=args.imap_host, imap_port=args.imap_port, imap_ssl=not args.no_ssl,
            )
            password = getpass.getpass("App password (hidden): ").replace(" ", "")
            print("Checking login…")
            try:
                GmailIMAP(address, password, account.imap_host, account.imap_port, account.imap_ssl).check_login()
                store.add(account, password)
            except (MailError, ValueError) as exc:
                print(f"{RED}{exc}{RESET}")
                return 1
            print(f"{GREEN}Connected '{account.name}' ({address}), read-only.{RESET}")
            return 0

        if not args.name:
            print(f"Usage: jarvis mail {args.action} NAME")
            return 2

        if args.action == "remove":
            if store.remove(args.name):
                print(f"Removed '{args.name}' and deleted its saved password from the keychain.")
                return 0
            print(f"No account named {args.name!r}.")
            return 1

        try:
            client = store.client(args.name)
            if args.action == "test":
                client.check_login()
                print(f"{GREEN}OK{RESET}   '{args.name}' logged in and opened the mailbox read-only.")
                return 0
            if args.action == "search":
                query = " ".join(args.query) or "in:inbox"
                msgs = client.search(query, limit=10)
                if not msgs:
                    print("No matches.")
                for m in msgs:
                    print(f"{CYAN}{m.summary_line()}{RESET}\n  {m.snippet()}\n")
                return 0
        except MailError as exc:
            print(f"{RED}{exc}{RESET}")
            return 1
        return 0
    finally:
        store.close()


def cmd_chat(cfg, llm, memory, index, accounts, resume: bool) -> int:
    conv = memory.latest_conversation() if resume else None
    agent = Agent(cfg, llm, memory, conversation_id=conv, index=index, accounts=accounts)
    print(f"Jarvis {__version__} · model {cfg.model} · conversation #{agent.conversation_id}")
    print(f"{DIM}Type /exit to quit, /think to toggle visible reasoning.{RESET}")
    show = cfg.show_thinking
    while True:
        try:
            text = input(f"\n{GREEN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/think":
            show = not show
            print(f"Visible reasoning {'on' if show else 'off'}.")
            continue
        print(f"{CYAN}jarvis>{RESET} ", end="")
        try:
            run_turn(agent, text, show)
        except LLMError as exc:
            print(f"\n{RED}Error: {exc}{RESET}")
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="Local AI assistant")
    parser.add_argument("--model", help="override the model for this run")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check Ollama and model setup")
    p_chat = sub.add_parser("chat", help="interactive chat")
    p_chat.add_argument("--resume", action="store_true", help="continue the last conversation")
    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question", nargs="+")
    sub.add_parser("history", help="list past conversations")
    sub.add_parser("notes", help="show saved notes")
    p_index = sub.add_parser("index", help="manage the document index")
    p_index.add_argument("action", choices=["add", "remove", "update", "list"])
    p_index.add_argument("paths", nargs="*")
    p_search = sub.add_parser("search", help="search indexed documents directly")
    p_search.add_argument("query", nargs="+")
    p_mail = sub.add_parser("mail", help="manage email accounts")
    p_mail.add_argument("action", choices=["add", "list", "test", "remove", "search"])
    p_mail.add_argument("name", nargs="?")
    p_mail.add_argument("query", nargs="*")
    p_mail.add_argument("--imap-host", default="imap.gmail.com", help=argparse.SUPPRESS)
    p_mail.add_argument("--imap-port", type=int, default=993, help=argparse.SUPPRESS)
    p_mail.add_argument("--no-ssl", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"{RED}Config error: {exc}{RESET}", file=sys.stderr)
        return 2
    if args.model:
        cfg.model = args.model

    command = args.command or "chat"
    llm = OllamaClient(cfg.ollama_host, timeout=cfg.request_timeout)
    try:
        if command == "doctor":
            return cmd_doctor(cfg, llm)
        if command == "index":
            if args.action in ("add", "remove") and not args.paths:
                print(f"Usage: jarvis index {args.action} FOLDER [FOLDER...]")
                return 2
            return cmd_index(cfg, llm, args)
        if command == "search":
            return cmd_search(cfg, llm, " ".join(args.query))
        if command == "mail":
            return cmd_mail(cfg, args)
        memory = Memory(cfg.db_path)
        index = open_index(cfg, llm)
        accounts = AccountStore(cfg.db_path)
        try:
            if command == "chat":
                return cmd_chat(cfg, llm, memory, index, accounts, getattr(args, "resume", False))
            if command == "ask":
                agent = Agent(cfg, llm, memory, index=index, accounts=accounts)
                try:
                    run_turn(agent, " ".join(args.question), cfg.show_thinking)
                except LLMError as exc:
                    print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
                    return 1
                return 0
            if command == "history":
                for r in memory.list_conversations():
                    print(f"#{r['id']:<4} {r['created_at']}  {r['n']:>3} msgs  {r['title'] or ''}")
                return 0
            if command == "notes":
                notes = memory.list_notes()
                if not notes:
                    print("No saved notes yet.")
                for n in notes:
                    print(f"#{n['id']:<4} {n['text']}")
                return 0
        finally:
            memory.close()
            index.close()
            accounts.close()
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
