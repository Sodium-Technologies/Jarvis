"""A small fake Gmail IMAP server (plain TCP) for exercising the real imaplib client.

Gmail quirks it imitates:
  * "All Mail" is found via the \\All flag and has a localized name here.
  * X-GM-RAW search with UTF-8 literals; X-GM-MSGID / X-GM-THRID / X-GM-LABELS.
  * FETCH items after the message literal (UID comes last), as Gmail may do.
It records every command so tests can assert nothing mutating is ever sent.
"""

from __future__ import annotations

import email
import email.policy
import re
import socketserver
import threading

ALL_MAIL = "[Gmail]/Tous les messages"


class FakeGmail:
    def __init__(self, user: str, password: str):
        self.user, self.password = user, password
        self.messages: list[dict] = []
        self.commands: list[str] = []

    def add(self, raw: bytes, *, thread: int | None = None, labels=("\\Inbox",)) -> int:
        uid = len(self.messages) + 1
        msgid = 1800000000000000000 + uid
        self.messages.append({"uid": uid, "msgid": msgid, "thrid": thread or msgid,
                              "labels": list(labels), "raw": raw})
        return msgid

    # -- X-GM-RAW: a tiny subset of Gmail search ------------------------------
    def matches(self, m: dict, query: str) -> bool:
        msg = email.message_from_bytes(m["raw"], policy=email.policy.default)
        names = [p.get_filename() or "" for p in msg.walk()]
        body = msg.get_body(preferencelist=("plain", "html"))
        text = (str(msg["Subject"]) + " " + (body.get_content() if body else "")).lower()
        for tok in query.lower().split():
            key, _, val = tok.partition(":")
            if not val:
                ok = tok in text or tok in str(msg["From"]).lower()
            elif key == "from":
                ok = val in str(msg["From"]).lower()
            elif key == "subject":
                ok = val in str(msg["Subject"]).lower()
            elif key == "has" and val == "attachment":
                ok = any(names)
            elif key == "filename":
                ok = any(val in n.lower() for n in names)
            else:  # newer_than:, in:, is: ... accepted, not filtered
                ok = True
            if not ok:
                return False
        return True


def _handler(fake: FakeGmail):
    class Handler(socketserver.StreamRequestHandler):
        def send(self, data: bytes | str):
            self.wfile.write(data.encode() if isinstance(data, str) else data)

        def read_command(self) -> tuple[str, bytes | None] | None:
            line = self.rfile.readline()
            if not line:
                return None
            line = line.rstrip(b"\r\n")
            literal = None
            m = re.search(rb"\{(\d+)\+?\}$", line)
            if m:
                self.send("+ go ahead\r\n")
                self.wfile.flush()
                literal = self.rfile.read(int(m.group(1)))
                self.rfile.readline()  # trailing CRLF
                line = line[: m.start()].rstrip()
            return line.decode(), literal

        def handle(self):
            self.send("* OK Gimap ready\r\n")
            selected = False
            while True:
                got = self.read_command()
                if got is None:
                    return
                line, literal = got
                fake.commands.append(line)
                tag, _, rest = line.partition(" ")
                cmd = rest.split(" ")[0].upper()
                if cmd == "CAPABILITY":
                    self.send(f"* CAPABILITY IMAP4rev1 X-GM-EXT-1 AUTH=PLAIN\r\n{tag} OK done\r\n")
                elif cmd == "LOGIN":
                    parts = re.findall(r'"((?:\\.|[^"\\])*)"|(\S+)', rest)[1:]
                    user, pw = [(a or b).replace('\\"', '"').replace("\\\\", "\\") for a, b in parts]
                    if (user, pw) == (fake.user, fake.password):
                        self.send(f"{tag} OK {user} authenticated (Success)\r\n")
                    else:
                        self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials (Failure)\r\n")
                elif cmd == "LIST":
                    self.send('* LIST (\\HasNoChildren) "/" "INBOX"\r\n'
                              f'* LIST (\\All \\HasNoChildren) "/" "{ALL_MAIL}"\r\n'
                              '* LIST (\\HasNoChildren \\Sent) "/" "[Gmail]/Sent Mail"\r\n'
                              f"{tag} OK Success\r\n")
                elif cmd in ("EXAMINE", "SELECT"):
                    selected = True
                    ro = "READ-ONLY" if cmd == "EXAMINE" else "READ-WRITE"
                    self.send(f"* {len(fake.messages)} EXISTS\r\n* OK [UIDVALIDITY 7] UIDs valid\r\n"
                              f"{tag} OK [{ro}] {ALL_MAIL} selected. (Success)\r\n")
                elif cmd == "UID" and selected:
                    self.uid_command(tag, rest[4:], literal)
                elif cmd == "LOGOUT":
                    self.send(f"* BYE LOGOUT Requested\r\n{tag} OK 73 good day (Success)\r\n")
                    return
                else:
                    self.send(f"{tag} BAD Unknown command\r\n")

        def uid_command(self, tag: str, args: str, literal: bytes | None):
            sub = args.split(" ")[0].upper()
            if sub == "SEARCH":
                if "X-GM-RAW" in args:
                    query = literal.decode("utf-8") if literal is not None else \
                        re.search(r'X-GM-RAW "(.*)"', args).group(1)
                    hits = [m["uid"] for m in fake.messages if fake.matches(m, query)]
                else:
                    key, val = re.search(r"(X-GM-MSGID|X-GM-THRID) (\d+)", args).groups()
                    field = "msgid" if key == "X-GM-MSGID" else "thrid"
                    hits = [m["uid"] for m in fake.messages if str(m[field]) == val]
                self.send(f"* SEARCH {' '.join(map(str, hits))}\r\n{tag} OK SEARCH completed (Success)\r\n")
            elif sub == "FETCH":
                uids = {int(u) for u in args.split(" ")[1].split(",")}
                partial = re.search(r"BODY\.PEEK\[\]<0\.(\d+)>", args)
                for seq, m in enumerate(fake.messages, 1):
                    if m["uid"] not in uids:
                        continue
                    raw = m["raw"][: int(partial.group(1))] if partial else m["raw"]
                    labels = " ".join(l if l.startswith("\\") else f'"{l}"' for l in m["labels"])
                    head = (f"* {seq} FETCH (X-GM-THRID {m['thrid']} X-GM-MSGID {m['msgid']} "
                            f"X-GM-LABELS ({labels}) RFC822.SIZE {len(m['raw'])} "
                            f"BODY[]{'<0>' if partial else ''} {{{len(raw)}}}\r\n")
                    self.send(head.encode() + raw + f" UID {m['uid']})\r\n".encode())
                self.send(f"{tag} OK Success\r\n")
            else:
                self.send(f"{tag} NO [CANNOT] not supported in fake\r\n")

    return Handler


class FakeIMAPServer:
    def __init__(self, fake: FakeGmail):
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _handler(fake))
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
