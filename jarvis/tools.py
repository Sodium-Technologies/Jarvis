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
