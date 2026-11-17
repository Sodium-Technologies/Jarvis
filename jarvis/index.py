"""Local file index: find documents by name, keyword or meaning.

Storage is a separate SQLite file (default ~/.jarvis/index.db):
  folders  the folders the user chose to index
  files    every file found (name-searchable even if its content is not)
  chunks   passages of extracted text, each with an optional embedding
  chunks_fts  SQLite full-text index over (file name, passage text)

Search combines keyword ranking (BM25) and embedding similarity using
reciprocal rank fusion, then returns the best passage per file.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .extract import ExtractError, can_extract, extract_text
from .llm import LLMError, OllamaClient

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS folders (path TEXT PRIMARY KEY, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    path    TEXT NOT NULL UNIQUE,
    folder  TEXT NOT NULL,
    name    TEXT NOT NULL,
    size    INTEGER NOT NULL,
    mtime   REAL NOT NULL,
    status  TEXT NOT NULL,          -- ok | name-only | cloud-only | error
    error   TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    seq       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(name, text, tokenize='unicode61');
"""

SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "$RECYCLE.BIN", "System Volume Information"}
SKIP_DIR_SUFFIXES = (".app", ".photoslibrary", ".musiclibrary", ".tvlibrary", ".bundle", ".framework")
SF_DATALESS = 0x40000000  # macOS: file content lives only in the cloud (iCloud/OneDrive/Drive)

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
EMBED_BATCH = 16

# Some embedding models are trained with task prefixes.
_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
}


def _prefixes(model: str) -> tuple[str, str]:
    base = model.split(":")[0]
    return _PREFIXES.get(base, ("", ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into passages of about `size` chars, breaking on paragraph/line ends."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _fts_query(query: str) -> str:
    terms = re.findall(r"\w+", query.lower())
    return " OR ".join(f'"{t}"' for t in terms)


def _snippet(text: str, query: str, width: int = 320) -> str:
    low = text.lower()
    pos = -1
    for term in sorted(re.findall(r"\w{3,}", query.lower()), key=len, reverse=True):
        pos = low.find(term)
        if pos >= 0:
            break
    start = max(0, pos - width // 3) if pos >= 0 else 0
    snip = " ".join(text[start : start + width].split())
    return ("…" if start > 0 else "") + snip + ("…" if start + width < len(text) else "")


@dataclass
class Hit:
    path: str
    name: str
    snippet: str
    modified: str
    score: float


@dataclass
class UpdateStats:
    scanned: int = 0
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    embedded: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    embed_warning: str | None = None


ProgressCallback = Callable[[str], None]


class FileIndex:
    def __init__(
        self,
        db_path: Path,
        llm: OllamaClient | None,
        embed_model: str = "",
        max_file_mb: float = 50.0,
        min_similarity: float = 0.45,
    ):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.llm = llm
        self.embed_model = embed_model
        self.max_bytes = int(max_file_mb * 1024 * 1024)
        self.min_similarity = min_similarity
        self._matrix: np.ndarray | None = None
        self._matrix_ids: np.ndarray | None = None
        self._reset_embeddings_if_model_changed()

    def close(self) -> None:
        self.db.close()

    # ---- folders ----------------------------------------------------------

    def add_folder(self, folder: str | Path) -> Path:
        path = Path(folder).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Not a folder: {path}")
        if path == Path(path.anchor):
            raise ValueError("Refusing to index a whole drive; choose specific folders.")
        self.db.execute("INSERT OR IGNORE INTO folders VALUES (?, ?)", (str(path), _now()))
        self.db.commit()
        return path

    def remove_folder(self, folder: str | Path) -> bool:
        path = str(Path(folder).expanduser().resolve())
        cur = self.db.execute("DELETE FROM folders WHERE path = ?", (path,))
        for row in self.db.execute("SELECT id FROM files WHERE folder = ?", (path,)).fetchall():
            self._delete_file(row["id"])
        self.db.commit()
        self._matrix = None
        return cur.rowcount > 0

    def folders(self) -> list[str]:
        return [r["path"] for r in self.db.execute("SELECT path FROM folders ORDER BY path")]

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        by_status = dict(self.db.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall())
        return {
            "files": q("SELECT COUNT(*) FROM files"),
            "chunks": q("SELECT COUNT(*) FROM chunks WHERE text != ''"),
            "embedded": q("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"),
            "by_status": by_status,
        }

    def is_allowed(self, path: str | Path) -> bool:
        """True if path is inside one of the indexed folders (blocks ../ escapes)."""
        try:
            real = Path(path).expanduser().resolve()
        except OSError:
            return False
        return any(real.is_relative_to(f) for f in self.folders())

    # ---- indexing ---------------------------------------------------------

    def update(self, progress: ProgressCallback | None = None) -> UpdateStats:
        stats = UpdateStats()
        for folder in self.folders():
            seen: set[str] = set()
            if not Path(folder).is_dir():
                stats.errors.append((folder, "folder is missing or not accessible"))
                continue
            for path in self._walk(Path(folder)):
                stats.scanned += 1
                seen.add(str(path))
                try:
                    changed = self._index_file(path, folder, stats)
                except OSError as exc:
                    stats.errors.append((str(path), str(exc)))
                    continue
                if changed:
                    stats.indexed += 1
                    if progress and stats.indexed % 100 == 0:
                        progress(f"indexed {stats.indexed} files…")
                else:
                    stats.unchanged += 1
            for row in self.db.execute("SELECT id, path FROM files WHERE folder = ?", (folder,)).fetchall():
                if row["path"] not in seen:
                    self._delete_file(row["id"])
                    stats.removed += 1
            self.db.commit()
        self._embed_pending(stats, progress)
        self._matrix = None
        return stats

    def _walk(self, root: Path):
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = sorted(
                d for d in dirnames
                if not d.startswith(".") and d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
            )
            for name in sorted(filenames):
                if name.startswith((".", "~$")) or name in ("Icon\r", "desktop.ini", "Thumbs.db"):
                    continue
                yield Path(dirpath) / name

    def _index_file(self, path: Path, folder: str, stats: UpdateStats) -> bool:
        st = path.stat()
        row = self.db.execute("SELECT id, size, mtime FROM files WHERE path = ?", (str(path),)).fetchone()
        if row and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
            return False
        if row:
            self._delete_file(row["id"])

        status, error, text = "name-only", None, ""
        if getattr(st, "st_flags", 0) & SF_DATALESS:
            status = "cloud-only"  # don't force a download of every online-only file
        elif can_extract(path):
            if st.st_size > self.max_bytes:
                error = f"larger than {self.max_bytes // (1024 * 1024)} MB"
            else:
                try:
                    text = extract_text(path)
                    status = "ok"
                except ExtractError as exc:
                    status, error = "error", str(exc)
                    stats.errors.append((str(path), error))

        cur = self.db.execute(
            "INSERT INTO files (path, folder, name, size, mtime, status, error) VALUES (?,?,?,?,?,?,?)",
            (str(path), folder, path.name, st.st_size, st.st_mtime, status, error),
        )
        file_id = cur.lastrowid
        # Every file gets at least one (possibly empty) row so its name is searchable.
        for seq, chunk in enumerate(chunk_text(text) or [""]):
            cid = self.db.execute(
                "INSERT INTO chunks (file_id, seq, text) VALUES (?, ?, ?)", (file_id, seq, chunk)
            ).lastrowid
            name_terms = re.sub(r"[_\-.]+", " ", path.stem) + " " + path.suffix.lstrip(".")
            self.db.execute("INSERT INTO chunks_fts (rowid, name, text) VALUES (?, ?, ?)", (cid, name_terms, chunk))
        return True

    def _delete_file(self, file_id: int) -> None:
        self.db.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,)
        )
        self.db.execute("DELETE FROM files WHERE id = ?", (file_id,))

    # ---- embeddings -------------------------------------------------------

    def _reset_embeddings_if_model_changed(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
        if row and row["value"] != self.embed_model:
            self.db.execute("UPDATE chunks SET embedding = NULL")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", (self.embed_model,))
        self.db.commit()

    def _embed_pending(self, stats: UpdateStats, progress: ProgressCallback | None) -> None:
        if not self.embed_model or self.llm is None:
            return
        doc_prefix, _ = _prefixes(self.embed_model)
        rows = self.db.execute(
            """SELECT c.id, c.text, f.name FROM chunks c JOIN files f ON f.id = c.file_id
               WHERE c.embedding IS NULL"""
        ).fetchall()
        for i in range(0, len(rows), EMBED_BATCH):
            batch = rows[i : i + EMBED_BATCH]
            # File name is included so name-only files (photos, scans) can be found too.
            inputs = [f"{doc_prefix}{r['name']}\n{r['text']}".rstrip() for r in batch]
            try:
                vectors = self.llm.embed(self.embed_model, inputs)
            except LLMError as exc:
                stats.embed_warning = (
                    f"Search by meaning is unavailable ({exc}). Keyword search still works. "
                    f"Fix with: ollama pull {self.embed_model}, then run: jarvis index update"
                )
                break
            for r, vec in zip(batch, vectors):
                v = np.asarray(vec, dtype=np.float32)
                v /= np.linalg.norm(v) or 1.0
                self.db.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (v.tobytes(), r["id"]))
            self.db.commit()
            stats.embedded += len(batch)
            if progress and ((i // EMBED_BATCH) % 10 == 9 or i + EMBED_BATCH >= len(rows)):
                progress(f"embedded {stats.embedded}/{len(rows)} passages…")

    def _load_matrix(self) -> None:
        rows = self.db.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        if not rows:
            self._matrix, self._matrix_ids = np.zeros((0, 0), np.float32), np.zeros(0, np.int64)
            return
        self._matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        self._matrix_ids = np.array([r["id"] for r in rows], dtype=np.int64)

    # ---- search -----------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        """Best-matching files for a natural-language or keyword query."""
        ranked: dict[int, float] = {}
        k = 60  # reciprocal rank fusion constant

        fts = _fts_query(query)
        if fts:
            rows = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 3.0, 1.0) LIMIT 50",
                (fts,),
            ).fetchall()
            for rank, r in enumerate(rows):
                ranked[r[0]] = ranked.get(r[0], 0) + 1 / (k + rank)

        if self.embed_model and self.llm is not None:
            if self._matrix is None:
                self._load_matrix()
            if self._matrix.size:
                try:
                    _, q_prefix = _prefixes(self.embed_model)
                    q = np.asarray(self.llm.embed(self.embed_model, [q_prefix + query])[0], dtype=np.float32)
                    q /= np.linalg.norm(q) or 1.0
                    if q.shape[0] == self._matrix.shape[1]:
                        sims = self._matrix @ q
                        # Similarity always returns *something*; drop weak matches so an
                        # unrelated query yields no results instead of random files.
                        top = [i for i in np.argsort(-sims)[:30] if sims[i] >= self.min_similarity]
                        for rank, idx in enumerate(top):
                            cid = int(self._matrix_ids[idx])
                            ranked[cid] = ranked.get(cid, 0) + 1 / (k + rank)
                except LLMError:
                    pass  # keyword results still stand

        best_per_file: dict[int, tuple[float, sqlite3.Row]] = {}
        if ranked:
            ids = sorted(ranked, key=ranked.get, reverse=True)[:200]
            marks = ",".join("?" * len(ids))
            rows = self.db.execute(
                f"""SELECT c.id, c.file_id, c.text, f.path, f.name, f.mtime FROM chunks c
                    JOIN files f ON f.id = c.file_id WHERE c.id IN ({marks})""",
                ids,
            ).fetchall()
            for r in rows:
                score = ranked[r["id"]]
                if r["file_id"] not in best_per_file or score > best_per_file[r["file_id"]][0]:
                    best_per_file[r["file_id"]] = (score, r)

        hits = []
        for score, r in sorted(best_per_file.values(), key=lambda x: x[0], reverse=True)[:limit]:
            hits.append(Hit(
                path=r["path"], name=r["name"],
                snippet=_snippet(r["text"], query) if r["text"] else "(file name match; content not indexed)",
                modified=datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d"),
                score=round(score, 4),
            ))
        return hits

    def find_files(self, pattern: str, limit: int = 30) -> list[sqlite3.Row]:
        """Find files by name.

        Wildcards (*.pdf, lease*2025*) match the whole name. Otherwise every word
        must appear in the name, ignoring case and _ - . separators, so
        "lease 4b" matches "Lease_4B_2025.pdf".
        """
        if any(ch in pattern for ch in "*?["):
            pat = pattern.lower()
            match = lambda name: fnmatch.fnmatchcase(name.lower(), pat)
        else:
            words = re.findall(r"\w+", pattern.lower())
            norm = lambda name: re.sub(r"[_\-.\s]+", " ", name.lower())
            match = lambda name: bool(words) and all(w in norm(name) for w in words)
        out = []
        for r in self.db.execute("SELECT path, name, size, mtime, status FROM files ORDER BY mtime DESC"):
            if match(r["name"]):
                out.append(r)
                if len(out) >= limit:
                    break
        return out
