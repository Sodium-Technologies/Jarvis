"""Turn files into plain text.

Each extractor returns the document's text, or raises ExtractError. Files
whose type has no extractor are still indexed by name, so they can be found.
"""

from __future__ import annotations

import csv
import email
import email.policy
import io
from html.parser import HTMLParser
from pathlib import Path


class ExtractError(Exception):
    pass


def _text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ExtractError("password-protected PDF") from exc
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[page {i}]\n{text}")
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"unreadable PDF: {exc}") from exc
    if not pages:
        # Scanned image PDFs have no text layer; OCR is a later step.
        raise ExtractError("no text layer (probably a scanned image)")
    return "\n\n".join(pages)


def _docx(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _xlsx(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    out = []
    try:
        for ws in wb.worksheets:
            out.append(f"[sheet {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                vals = ["" if v is None else str(v) for v in row]
                if any(vals):
                    out.append(" | ".join(vals).rstrip(" |"))
    finally:
        wb.close()
    return "\n".join(out)


def _csv(path: Path) -> str:
    text = _text(path)
    rows = csv.reader(io.StringIO(text))
    return "\n".join(" | ".join(r) for r in rows if any(c.strip() for c in r))


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    lines = (" ".join(l.split()) for l in "".join(p.parts).splitlines())
    return "\n".join(l for l in lines if l)


def _html(path: Path) -> str:
    return html_to_text(_text(path))


def _eml(path: Path) -> str:
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    header = "\n".join(
        f"{h}: {msg[h]}" for h in ("From", "To", "Cc", "Date", "Subject") if msg[h]
    )
    body = msg.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        text = body.get_content()
        if body.get_content_type() == "text/html":
            text = html_to_text(text)
    attachments = [a.get_filename() for a in msg.iter_attachments() if a.get_filename()]
    if attachments:
        header += "\nAttachments: " + ", ".join(attachments)
    return f"{header}\n\n{text}"


EXTRACTORS = {
    ".txt": _text, ".md": _text, ".log": _text, ".json": _text, ".xml": _text,
    ".pdf": _pdf,
    ".docx": _docx,
    ".xlsx": _xlsx, ".xlsm": _xlsx,
    ".csv": _csv,
    ".html": _html, ".htm": _html,
    ".eml": _eml,
}


def can_extract(path: Path) -> bool:
    return path.suffix.lower() in EXTRACTORS


def extract_text(path: Path) -> str:
    fn = EXTRACTORS.get(path.suffix.lower())
    if fn is None:
        raise ExtractError(f"unsupported file type {path.suffix or '(none)'}")
    try:
        return fn(path)
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"{type(exc).__name__}: {exc}") from exc
