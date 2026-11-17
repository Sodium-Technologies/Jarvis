import os
import time
from email.message import EmailMessage
from pathlib import Path

import docx
import openpyxl
import pytest

from jarvis.agent import Agent
from jarvis.cli import main
from jarvis.config import load_config
from jarvis.extract import extract_text
from jarvis.index import FileIndex, chunk_text
from jarvis.llm import OllamaClient
from jarvis.memory import Memory


def make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal valid PDF with one line of text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + (o if isinstance(o, bytes) else o.encode()) + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(bytes(out))


@pytest.fixture
def docs(tmp_path) -> Path:
    root = tmp_path / "Documents"
    (root / "Black Squirrel Flats").mkdir(parents=True)
    (root / "LUXE").mkdir()

    make_pdf(root / "LUXE" / "Lease_Unit_4B_2025.pdf",
             ["Residential lease between LUXE MANAGEMENT LLC and tenant Maria Lopez.",
              "Monthly rent is 2150 dollars due on the first."])

    d = docx.Document()
    d.add_paragraph("Plumbing repair at 1412 Elm St: kitchen sink drain replaced.")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Total", "$385.00"
    d.save(root / "Black Squirrel Flats" / "Work order 0921.docx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "October"
    ws.append(["Guest", "Nights", "Occupancy tax"])
    ws.append(["Chen", 3, 42.15])
    wb.save(root / "Black Squirrel Flats" / "Occupancy report Oct.xlsx")

    (root / "Black Squirrel Flats" / "deposits.csv").write_text("date,amount,payer\n2025-10-01,1200,Unit 2\n")

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "owner@example.com", "naveed@example.com", "Q3 owner statement"
    msg["Date"] = "Tue, 14 Oct 2025 10:00:00 -0400"
    msg.set_content("Please send the reconciled owner statement for Q3.")
    msg.add_attachment(b"x", maintype="application", subtype="pdf", filename="statement.pdf")
    (root / "Black Squirrel Flats" / "owner statement request.eml").write_bytes(bytes(msg))

    (root / "LUXE" / "building photo.jpg").write_bytes(b"\xff\xd8\xff")      # name-only
    make_pdf(root / "LUXE" / "scanned notice.pdf", [""])                       # no text layer
    (root / ".hidden.txt").write_text("secret")                               # skipped
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.txt").write_text("lease lease lease")      # skipped
    (root / "~$Work order 0921.docx").write_text("lock")                      # Office lock file
    return root


def open_index(env, embed_model="nomic-embed-text") -> FileIndex:
    cfg = load_config()
    # The fake bag-of-words embeddings score lower than real models, so use a lower floor.
    return FileIndex(cfg.index_path, OllamaClient(cfg.ollama_host), embed_model, min_similarity=0.2)


# ---- extraction -------------------------------------------------------------


def test_extractors(docs):
    pdf = extract_text(docs / "LUXE" / "Lease_Unit_4B_2025.pdf")
    assert "[page 2]" in pdf and "2150" in pdf
    assert "$385.00" in extract_text(docs / "Black Squirrel Flats" / "Work order 0921.docx")
    xlsx = extract_text(docs / "Black Squirrel Flats" / "Occupancy report Oct.xlsx")
    assert "[sheet October]" in xlsx and "Chen | 3 | 42.15" in xlsx
    eml = extract_text(docs / "Black Squirrel Flats" / "owner statement request.eml")
    assert "Subject: Q3 owner statement" in eml and "Attachments: statement.pdf" in eml


def test_chunking_overlaps_and_covers_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30))
    chunks = chunk_text(text)
    assert len(chunks) > 5 and all(len(c) <= 1200 for c in chunks)
    assert "Paragraph 0." in chunks[0] and "Paragraph 29." in chunks[-1]


# ---- indexing ---------------------------------------------------------------


def test_index_update_is_incremental(jarvis_env, docs):
    idx = open_index(jarvis_env)
    idx.add_folder(docs)
    s = idx.update()
    assert s.scanned == 7 and s.indexed == 7
    assert [e[0].endswith("scanned notice.pdf") for e in s.errors] == [True]
    st = idx.stats()
    assert st["by_status"] == {"ok": 5, "name-only": 1, "error": 1}
    # Every passage is embedded, plus the name-only photo (by file name).
    assert st["embedded"] == st["chunks"] + 2 and st["chunks"] > 0

    calls = jarvis_env.embed_calls
    s = idx.update()
    assert s.indexed == 0 and s.unchanged == 7 and jarvis_env.embed_calls == calls

    # Edit one file, delete another.
    csv = docs / "Black Squirrel Flats" / "deposits.csv"
    csv.write_text("date,amount,payer\n2025-11-01,1350,Unit 5\n")
    os.utime(csv, (time.time() + 5, time.time() + 5))
    (docs / "LUXE" / "building photo.jpg").unlink()
    s = idx.update()
    assert (s.indexed, s.removed) == (1, 1)
    assert idx.search("1350")[0].name == "deposits.csv"
    assert idx.search("1200") == []


def test_keyword_and_name_search(jarvis_env, docs):
    idx = open_index(jarvis_env)
    idx.add_folder(docs)
    idx.update()
    top = idx.search("occupancy tax October")[0]
    assert top.name == "Occupancy report Oct.xlsx"
    assert idx.search("building photo")[0].name == "building photo.jpg"

    names = [r["name"] for r in idx.find_files("lease 4b")]
    assert names == ["Lease_Unit_4B_2025.pdf"]
    assert {r["name"] for r in idx.find_files("*.pdf")} == {"Lease_Unit_4B_2025.pdf", "scanned notice.pdf"}


def test_meaning_search_finds_documents_without_shared_keywords(jarvis_env, docs):
    idx = open_index(jarvis_env)
    idx.add_folder(docs)
    idx.update()
    # No document contains "tenancy" or "contract"; the lease is found by meaning.
    assert idx.search("tenancy contract")[0].name == "Lease_Unit_4B_2025.pdf"
    assert idx.search("pipe leak")[0].name == "Work order 0921.docx"


def test_works_without_embedding_model(jarvis_env, docs):
    jarvis_env.models = ["qwen3:8b"]
    idx = open_index(jarvis_env)
    idx.add_folder(docs)
    s = idx.update()
    assert s.embed_warning and "ollama pull nomic-embed-text" in s.embed_warning
    assert idx.search("2150 rent")[0].name == "Lease_Unit_4B_2025.pdf"  # keywords still work

    jarvis_env.models.append("nomic-embed-text:latest")  # user pulls the model
    s = idx.update()
    assert s.indexed == 0 and s.embedded > 0 and not s.embed_warning


def test_changing_embedding_model_reembeds(jarvis_env, docs):
    idx = open_index(jarvis_env)
    idx.add_folder(docs)
    idx.update()
    idx.close()
    jarvis_env.models.append("mxbai-embed-large:latest")
    idx = open_index(jarvis_env, embed_model="mxbai-embed-large")
    assert idx.stats()["embedded"] == 0
    assert idx.update().embedded > 0


def test_remove_folder_and_root_refused(jarvis_env, docs):
    idx = open_index(jarvis_env)
    with pytest.raises(ValueError):
        idx.add_folder("/")
    idx.add_folder(docs)
    idx.update()
    assert idx.remove_folder(docs)
    assert idx.stats()["files"] == 0 and idx.search("lease") == []


# ---- agent tools ------------------------------------------------------------


def test_agent_searches_and_reads_files(jarvis_env, docs):
    cfg = load_config()
    llm = OllamaClient(cfg.ollama_host)
    idx = FileIndex(cfg.index_path, llm, cfg.embed_model)
    idx.add_folder(docs)
    idx.update()
    agent = Agent(cfg, llm, Memory(cfg.db_path), index=idx)
    assert {"search_documents", "find_files", "read_file"} <= set(agent.tools.names())

    lease = str(docs / "LUXE" / "Lease_Unit_4B_2025.pdf")
    jarvis_env.queue_tool_call("search_documents", {"query": "rent for unit 4B"})
    jarvis_env.queue_tool_call("read_file", {"path": lease})
    jarvis_env.queue_tool_call("read_file", {"path": "/etc/passwd"})
    jarvis_env.queue_tool_call("read_file", {"path": str(docs / "LUXE" / ".." / ".." / "x.txt")})
    jarvis_env.queue_text("Rent is $2,150 (Lease_Unit_4B_2025.pdf).")
    agent.ask("What is the rent for unit 4B?")

    outputs = [m["content"] for m in jarvis_env.requests[-1]["messages"] if m["role"] == "tool"]
    assert lease in outputs[0]
    assert "Monthly rent is 2150" in outputs[1]
    assert "outside the folders" in outputs[2] and "outside the folders" in outputs[3]


def test_no_file_tools_until_a_folder_is_indexed(jarvis_env):
    cfg = load_config()
    llm = OllamaClient(cfg.ollama_host)
    agent = Agent(cfg, llm, Memory(cfg.db_path), index=FileIndex(cfg.index_path, llm, cfg.embed_model))
    assert "search_documents" not in agent.tools.names()


# ---- CLI --------------------------------------------------------------------


def test_cli_index_and_search(jarvis_env, docs, capsys):
    assert main(["index", "add", str(docs)]) == 0
    out = capsys.readouterr().out
    assert "7 files scanned" in out and "scanned notice.pdf: no text layer" in out

    assert main(["index", "list"]) == 0
    assert "5 ok" in capsys.readouterr().out

    assert main(["search", "owner", "statement"]) == 0
    assert "owner statement request.eml" in capsys.readouterr().out

    assert main(["index", "remove", str(docs)]) == 0
    assert main(["index", "update"]) == 1
