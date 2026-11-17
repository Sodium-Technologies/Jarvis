"""Read-only Gmail access over IMAP, using an app password.

Why IMAP + app password instead of the Gmail API: for personal @gmail.com
accounts an unverified Google Cloud app's sign-in expires every 7 days, while
an app password keeps working until you revoke it. Google's own search syntax
(from:, has:attachment, newer_than:7d, ...) is available through X-GM-RAW.

Read-only is enforced here, not just promised: the mailbox is opened with
EXAMINE (read-only) and messages are fetched with BODY.PEEK, so Gmail never
marks anything as read, and no command that changes mail exists in this class.
"""

from __future__ import annotations

import imaplib
import re
import socket
from contextlib import contextmanager

from .parse import MailMessage, get_attachment, parse_message

PREVIEW_BYTES = 32 * 1024  # enough for headers and the text body of most mail
MAX_FULL_BYTES = 30 * 1024 * 1024


class MailError(RuntimeError):
    pass


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


_LIST_RE = re.compile(rb'\((?P<flags>[^)]*)\) (?:"[^"]*"|NIL) (?P<name>.+)$')


class GmailIMAP:
    def __init__(self, address: str, password: str, host: str = "imap.gmail.com",
                 port: int = 993, ssl: bool = True, timeout: float = 30.0):
        self.address = address
        self.password = password.replace(" ", "")  # Google shows app passwords in groups of 4
        self.host, self.port, self.ssl, self.timeout = host, port, ssl, timeout

    # ---- connection -------------------------------------------------------

    @contextmanager
    def _session(self):
        try:
            cls = imaplib.IMAP4_SSL if self.ssl else imaplib.IMAP4
            imap = cls(self.host, self.port, timeout=self.timeout)
        except (OSError, socket.timeout) as exc:
            raise MailError(f"Cannot connect to {self.host}: {exc}") from exc
        try:
            try:
                imap.login(self.address, self.password)
            except imaplib.IMAP4.error as exc:
                raise MailError(
                    f"Gmail rejected the login for {self.address}. Check the app password "
                    "(Google Account → Security → 2-Step Verification → App passwords)."
                ) from exc
            mailbox = self._all_mail(imap)
            typ, _ = imap.select(_quote(mailbox), readonly=True)  # EXAMINE
            if typ != "OK":
                raise MailError(f"Cannot open mailbox {mailbox!r}")
            yield imap
        except imaplib.IMAP4.error as exc:
            raise MailError(f"IMAP error: {exc}") from exc
        except (OSError, socket.timeout) as exc:
            raise MailError(f"Connection to {self.host} failed: {exc}") from exc
        finally:
            try:
                imap.logout()
            except Exception:
                pass

    @staticmethod
    def _all_mail(imap) -> str:
        """Find Gmail's All Mail folder by its \\All flag (its name is localized)."""
        typ, data = imap.list()
        if typ == "OK":
            for line in data:
                if not isinstance(line, bytes):
                    continue
                m = _LIST_RE.search(line)
                if m and b"\\all" in m["flags"].lower():
                    return m["name"].decode("utf-8", "replace").strip().strip('"')
        return "INBOX"

    def check_login(self) -> None:
        with self._session():
            pass

    # ---- reading ----------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[MailMessage]:
        """Newest-first messages matching a Gmail search query."""
        with self._session() as imap:
            uids = self._search_uids(imap, "X-GM-RAW", query)
            uids = uids[-limit:][::-1]
            if not uids:
                return []
            msgs = self._fetch(imap, uids, full=False)
        order = {u: i for i, u in enumerate(uids)}  # newest first
        return [m for _, m in sorted(msgs, key=lambda x: order.get(x[0], len(order)))]

    def get(self, message_id: str) -> tuple[MailMessage, bytes]:
        """Full message (and its raw bytes) by Gmail message id."""
        if not str(message_id).isdigit():
            raise MailError(f"Invalid message id {message_id!r}")
        with self._session() as imap:
            uids = self._search_uids(imap, "X-GM-MSGID", str(message_id))
            if not uids:
                raise MailError(f"No message with id {message_id}")
            (_, msg, raw), = self._fetch(imap, uids[:1], full=True, keep_raw=True)
        return msg, raw

    def thread(self, thread_id: str, limit: int = 20) -> list[MailMessage]:
        if not str(thread_id).isdigit():
            raise MailError(f"Invalid thread id {thread_id!r}")
        with self._session() as imap:
            uids = self._search_uids(imap, "X-GM-THRID", str(thread_id))[-limit:]
            msgs = self._fetch(imap, uids, full=True) if uids else []
        return [m for _, m in sorted(msgs, key=lambda x: x[0])]

    def attachment(self, message_id: str, filename: str) -> tuple[str, bytes]:
        _, raw = self.get(message_id)
        found = get_attachment(raw, filename)
        if not found:
            raise MailError(f"Message {message_id} has no attachment named {filename!r}")
        return found

    # ---- IMAP plumbing ----------------------------------------------------

    @staticmethod
    def _search_uids(imap, key: str, value: str) -> list[int]:
        if key == "X-GM-RAW":
            # Send the query as a UTF-8 literal so any characters are safe.
            imap.literal = value.encode("utf-8")
            typ, data = imap.uid("SEARCH", "CHARSET", "UTF-8", key)
        else:
            typ, data = imap.uid("SEARCH", key, value)
        if typ != "OK":
            raise MailError(f"Search failed: {data!r}")
        return [int(x) for x in b" ".join(d for d in data if d).split()]

    @staticmethod
    def _fetch(imap, uids: list[int], *, full: bool, keep_raw: bool = False):
        body = "BODY.PEEK[]" if full else f"BODY.PEEK[]<0.{PREVIEW_BYTES}>"
        items = f"(UID X-GM-MSGID X-GM-THRID X-GM-LABELS RFC822.SIZE {body})"
        typ, data = imap.uid("FETCH", ",".join(map(str, uids)), items)
        if typ != "OK":
            raise MailError(f"Fetch failed: {data!r}")

        # imaplib returns (meta, literal) tuples; Gmail may put items after the
        # literal, which then arrive as the following bytes element.
        records: list[list] = []
        for part in data:
            if isinstance(part, tuple):
                records.append([part[0], part[1]])
            elif isinstance(part, bytes) and records:
                records[-1][0] += b" " + part

        out = []
        for meta, raw in records:
            def num(name: bytes) -> str:
                m = re.search(name + rb" (\d+)", meta)
                return m.group(1).decode() if m else ""
            labels_m = re.search(rb"X-GM-LABELS \(([^)]*)\)", meta)
            labels = re.findall(r'"([^"]*)"|(\S+)', labels_m.group(1).decode("utf-8", "replace")) if labels_m else []
            labels = [a or b for a, b in labels]
            size = int(num(b"RFC822.SIZE") or 0)
            msg = parse_message(
                raw, id=num(b"X-GM-MSGID"), thread_id=num(b"X-GM-THRID"), labels=labels,
                size=size, truncated=not full and size > len(raw),
            )
            uid = int(num(b"UID") or 0)
            out.append((uid, msg, raw) if keep_raw else (uid, msg))
        return out
