"""Local persistence: conversation history and long-term notes.

Everything is stored in a single SQLite file under the Jarvis home directory
(default ~/.jarvis/jarvis.db). Nothing is sent anywhere.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    thinking        TEXT,
    tool_name       TEXT,
    tool_calls      TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---- conversations ---------------------------------------------------

    def new_conversation(self, title: str | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, _now())
        )
        self.db.commit()
        return cur.lastrowid

    def set_title(self, conversation_id: int, title: str) -> None:
        self.db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        self.db.commit()

    def list_conversations(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT c.id, c.title, c.created_at, COUNT(m.id) AS n
               FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
               GROUP BY c.id ORDER BY c.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def latest_conversation(self) -> int | None:
        row = self.db.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def add_message(self, conversation_id: int, message: dict) -> None:
        self.db.execute(
            """INSERT INTO messages
               (conversation_id, role, content, thinking, tool_name, tool_calls, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                message["role"],
                message.get("content", ""),
                message.get("thinking"),
                message.get("tool_name"),
                json.dumps(message["tool_calls"]) if message.get("tool_calls") else None,
                _now(),
            ),
        )
        self.db.commit()

    def get_messages(self, conversation_id: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
        out = []
        for r in rows:
            msg: dict = {"role": r["role"], "content": r["content"]}
            if r["tool_name"]:
                msg["tool_name"] = r["tool_name"]
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            # Past reasoning is not sent back to the model; it only wastes context.
            out.append(msg)
        return out

    # ---- long-term notes ("learning") -----------------------------------

    def add_note(self, text: str) -> int:
        cur = self.db.execute("INSERT INTO notes (text, created_at) VALUES (?, ?)", (text, _now()))
        self.db.commit()
        return cur.lastrowid

    def list_notes(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT id, text, created_at FROM notes ORDER BY id").fetchall()

    def delete_note(self, note_id: int) -> bool:
        cur = self.db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.db.commit()
        return cur.rowcount > 0
