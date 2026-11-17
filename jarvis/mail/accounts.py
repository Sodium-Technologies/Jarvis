"""Mail account profiles.

Each account records who the user is *for that account* (name, role and
organization), so Jarvis keeps client identities separate. Passwords are kept
in the operating system's keychain (macOS Keychain, Windows Credential Manager),
never in the database or config files.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .gmail import GmailIMAP, MailError

SERVICE = "jarvis-mail"

SCHEMA = """
CREATE TABLE IF NOT EXISTS mail_accounts (
    name         TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    address      TEXT NOT NULL,
    owner_name   TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL DEFAULT '',
    organization TEXT NOT NULL DEFAULT '',
    imap_host    TEXT NOT NULL,
    imap_port    INTEGER NOT NULL,
    imap_ssl     INTEGER NOT NULL DEFAULT 1,
    added_at     TEXT NOT NULL
);
"""


@dataclass
class MailAccount:
    name: str
    provider: str
    address: str
    owner_name: str = ""
    role: str = ""
    organization: str = ""
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_ssl: bool = True

    def describe(self) -> str:
        who = self.owner_name or "the user"
        parts = [f"'{self.name}': {self.address}"]
        if self.organization or self.role:
            parts.append(
                f"the user acts as {who}"
                + (f", {self.role}" if self.role else "")
                + (f" for {self.organization}" if self.organization else "")
            )
        elif self.owner_name:
            parts.append(f"the user's own account, as {self.owner_name}")
        return " — ".join(parts)


class KeyringSecrets:
    """Thin wrapper so tests can swap in an in-memory store."""

    def get(self, key: str) -> str | None:
        import keyring
        return keyring.get_password(SERVICE, key)

    def set(self, key: str, value: str) -> None:
        import keyring
        from keyring.errors import KeyringError, NoKeyringError
        try:
            keyring.set_password(SERVICE, key, value)
        except (NoKeyringError, KeyringError) as exc:
            raise MailError(f"Could not save the password to the system keychain: {exc}") from exc

    def delete(self, key: str) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError
        try:
            keyring.delete_password(SERVICE, key)
        except PasswordDeleteError:
            pass


class AccountStore:
    def __init__(self, db_path: Path, secrets=None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.secrets = secrets or KeyringSecrets()

    def close(self) -> None:
        self.db.close()

    def add(self, account: MailAccount, password: str) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,30}", account.name):
            raise ValueError("Account name must be short, lowercase letters/numbers/-/_ (e.g. 'bsf').")
        if self.get(account.name):
            raise ValueError(f"An account named {account.name!r} already exists.")
        self.secrets.set(self._key(account), password.replace(" ", ""))
        d = asdict(account)
        self.db.execute(
            f"INSERT INTO mail_accounts ({', '.join(d)}, added_at) VALUES ({', '.join('?' * len(d))}, ?)",
            (*d.values(), datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.db.commit()

    def remove(self, name: str) -> bool:
        acct = self.get(name)
        if not acct:
            return False
        self.secrets.delete(self._key(acct))
        self.db.execute("DELETE FROM mail_accounts WHERE name = ?", (name,))
        self.db.commit()
        return True

    def get(self, name: str) -> MailAccount | None:
        row = self.db.execute("SELECT * FROM mail_accounts WHERE name = ?", (name,)).fetchone()
        return self._from_row(row) if row else None

    def list(self) -> list[MailAccount]:
        return [self._from_row(r) for r in self.db.execute("SELECT * FROM mail_accounts ORDER BY name")]

    def client(self, name: str) -> GmailIMAP:
        acct = self.get(name)
        if not acct:
            names = ", ".join(a.name for a in self.list()) or "none"
            raise MailError(f"No mail account named {name!r}. Accounts: {names}")
        password = self.secrets.get(self._key(acct))
        if not password:
            raise MailError(f"No saved password for {acct.address}. Re-add it: jarvis mail add")
        if acct.provider != "gmail":
            raise MailError(f"Provider {acct.provider!r} is not supported yet.")
        return GmailIMAP(acct.address, password, acct.imap_host, acct.imap_port, acct.imap_ssl)

    @staticmethod
    def _key(acct: MailAccount) -> str:
        return f"{acct.name}:{acct.address}"

    @staticmethod
    def _from_row(row) -> MailAccount:
        return MailAccount(
            name=row["name"], provider=row["provider"], address=row["address"],
            owner_name=row["owner_name"], role=row["role"], organization=row["organization"],
            imap_host=row["imap_host"], imap_port=row["imap_port"], imap_ssl=bool(row["imap_ssl"]),
        )
