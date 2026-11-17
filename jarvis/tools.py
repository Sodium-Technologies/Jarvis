"""Tool registry.

A tool is a plain Python function plus a JSON schema the model sees. Tools that
change the outside world (send an email, delete a file, ...) must be registered
with confirm=True: the agent then asks the user before running them, every time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from pathlib import Path

from .extract import ExtractError, extract_text
from .memory import Memory


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]
    confirm: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Returns True if the user approves running the tool with these arguments.
ConfirmCallback = Callable[[str, dict], bool]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def run(self, name: str, arguments: dict, confirm: ConfirmCallback | None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: there is no tool named {name!r}. Available: {', '.join(self._tools)}"
        if tool.confirm:
            if confirm is None or not confirm(name, arguments):
                return "The user declined this action. Do not retry it; ask what they want instead."
        try:
            result = tool.func(**arguments)
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # report tool failures to the model instead of crashing
            return f"Error while running {name}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)


def builtin_tools(memory: Memory) -> list[Tool]:
    def get_current_datetime() -> str:
        return datetime.now().astimezone().strftime("%A, %d %B %Y, %H:%M %Z (UTC%z)")

    def remember(fact: str) -> str:
        note_id = memory.add_note(fact.strip())
        return f"Saved as note #{note_id}."

    def forget(note_id: int) -> str:
        return "Deleted." if memory.delete_note(int(note_id)) else f"No note #{note_id}."

    return [
        Tool(
            "get_current_datetime",
            "Get the current local date, time and time zone.",
            {"type": "object", "properties": {}},
            get_current_datetime,
        ),
        Tool(
            "remember",
            "Save a lasting fact or preference about the user or their work, so it is "
            "available in future conversations. Use when the user asks you to remember "
            "something or states a stable preference.",
            {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact, one sentence."}},
                "required": ["fact"],
            },
            remember,
        ),
        Tool(
            "forget",
            "Delete a saved note by its number.",
            {
                "type": "object",
                "properties": {"note_id": {"type": "integer"}},
                "required": ["note_id"],
            },
            forget,
            confirm=True,
        ),
    ]


def file_tools(index) -> list[Tool]:
    """Tools for finding and reading the user's indexed documents."""

    def search_documents(query: str, limit: int = 5) -> str:
        hits = index.search(query, limit=max(1, min(int(limit), 10)))
        if not hits:
            return "No matching documents found in the indexed folders."
        return "\n\n".join(
            f"[{i}] {h.path}  (modified {h.modified})\n{h.snippet}" for i, h in enumerate(hits, 1)
        )

    def find_files(name: str) -> str:
        rows = index.find_files(name)
        if not rows:
            return f"No files with a name matching {name!r}."
        return "\n".join(
            f"{r['path']}  ({r['size'] // 1024} KB, modified "
            f"{datetime.fromtimestamp(r['mtime']):%Y-%m-%d}{', ' + r['status'] if r['status'] != 'ok' else ''})"
            for r in rows
        )

    def read_file(path: str, offset: int = 0, max_chars: int = 6000) -> str:
        if not index.is_allowed(path):
            return "Error: that file is outside the folders Jarvis is allowed to read."
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Error: file not found: {path}"
        try:
            text = extract_text(p)
        except ExtractError as exc:
            return f"Error: cannot read {p.name}: {exc}"
        offset, max_chars = max(0, int(offset)), max(500, min(int(max_chars), 20000))
        part = text[offset : offset + max_chars]
        end = offset + len(part)
        more = f"\n\n[showing characters {offset}-{end} of {len(text)}; call again with offset={end} for more]" if end < len(text) else ""
        return f"{p}\n\n{part}{more}"

    return [
        Tool(
            "search_documents",
            "Search the user's indexed documents (PDF, Word, Excel, CSV, text, saved emails) by "
            "topic or keywords. Returns file paths with matching passages. Use this before "
            "answering questions about the user's files, leases, invoices, statements, etc.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in plain words."},
                    "limit": {"type": "integer", "description": "Max files to return (default 5)."},
                },
                "required": ["query"],
            },
            search_documents,
        ),
        Tool(
            "find_files",
            "Find files by name. Accepts part of a name ('lease 4B') or a wildcard ('*.xlsx', "
            "'*invoice*2025*'). Use when the user asks where a file is.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            find_files,
        ),
        Tool(
            "read_file",
            "Read the text of a file found by search_documents or find_files. Long files are "
            "returned in parts; use offset to continue.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path exactly as returned by a search."},
                    "offset": {"type": "integer", "description": "Character offset to start at."},
                },
                "required": ["path"],
            },
            read_file,
        ),
    ]
