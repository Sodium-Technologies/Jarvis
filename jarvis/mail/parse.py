"""Turn raw RFC 822 bytes into a simple, model-friendly message."""

from __future__ import annotations

import email
import email.policy
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime

from ..extract import html_to_text


@dataclass
class Attachment:
    filename: str
    content_type: str
    size: int


@dataclass
class MailMessage:
    id: str              # stable provider id (Gmail: X-GM-MSGID)
    thread_id: str
    date: str            # "YYYY-MM-DD HH:MM" local time, or "" if unknown
    sender: str
    to: str
    cc: str
    subject: str
    body: str
    labels: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    size: int = 0
    truncated: bool = False  # parsed from a partial download

    def snippet(self, n: int = 200) -> str:
        text = " ".join(l for l in self.body.splitlines() if not l.startswith(">"))
        text = " ".join(text.split())
        return text[:n] + ("…" if len(text) > n else "")

    def summary_line(self) -> str:
        att = f"  📎 {', '.join(a.filename for a in self.attachments)}" if self.attachments else ""
        return f"[id {self.id}] {self.date}  From: {self.sender}\n  Subject: {self.subject}{att}"


def _addr(value) -> str:
    if not value:
        return ""
    pairs = getaddresses([str(value)])
    return ", ".join(f"{n} <{a}>" if n else a for n, a in pairs if a or n)


def _date(value) -> str:
    try:
        return parsedate_to_datetime(str(value)).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, IndexError):
        return ""


def parse_message(raw: bytes, *, id: str = "", thread_id: str = "", labels=None,
                  size: int = 0, truncated: bool = False) -> MailMessage:
    msg: EmailMessage = email.message_from_bytes(raw, policy=email.policy.default)

    body = ""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body = part.get_content()
            if part.get_content_type() == "text/html":
                body = html_to_text(body)
    except (KeyError, LookupError, ValueError, AssertionError):
        # Broken or truncated MIME; fall back to whatever text is there.
        payload = msg.get_payload(decode=True)
        body = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else ""

    attachments = []
    for part in msg.walk():
        filename = part.get_filename()
        if filename and not part.is_multipart():
            try:
                payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = b""
            attachments.append(Attachment(filename, part.get_content_type(), len(payload)))

    return MailMessage(
        id=id, thread_id=thread_id, date=_date(msg["Date"]),
        sender=_addr(msg["From"]), to=_addr(msg["To"]), cc=_addr(msg["Cc"]),
        subject=str(msg["Subject"] or "(no subject)"),
        body=body.strip(), labels=list(labels or []), attachments=attachments,
        size=size or len(raw), truncated=truncated,
    )


def get_attachment(raw: bytes, filename: str) -> tuple[str, bytes] | None:
    """Return (filename, bytes) of the first attachment matching filename (case-insensitive)."""
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    wanted = filename.strip().lower()
    for part in msg.walk():
        name = part.get_filename()
        if name and name.lower() == wanted:
            return name, part.get_payload(decode=True) or b""
    return None
