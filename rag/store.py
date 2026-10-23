import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_sections
from .loaders import SUPPORTED, load

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id      INTEGER PRIMARY KEY,
    path    TEXT UNIQUE NOT NULL,
    title   TEXT NOT NULL,
    sha256  TEXT NOT NULL,
    chunks  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading   TEXT NOT NULL,
    text      TEXT NOT NULL,
    page      INTEGER,
    position  INTEGER NOT NULL,
    embedding BLOB
);
"""


@dataclass
class StoredChunk:
    id: int
    doc_title: str
    doc_path: str
    heading: str
    text: str
    page: int | None
    embedding: np.ndarray | None

    @property
    def indexed_text(self) -> str:
        return f"{self.doc_title} – {self.heading}\n{self.text}"

    def citation(self) -> str:
        where = f", p. {self.page}" if self.page else ""
        return f"{self.doc_title} › {self.heading}{where}"


class Store:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def ingest(self, path: Path, embedder=None) -> str:
        """Add or refresh one file. Returns 'added', 'updated' or 'unchanged'."""
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = self.conn.execute("SELECT id, sha256 FROM documents WHERE path = ?", (str(path),)).fetchone()
        if row and row["sha256"] == digest:
            return "unchanged"

        title, sections = load(path)
        chunks = chunk_sections(title, sections)
        vectors = embedder.embed([c.indexed_text for c in chunks]) if (embedder and chunks) else None

        with self.conn:
            if row:
                self.conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            doc_id = self.conn.execute(
                "INSERT INTO documents (path, title, sha256, chunks) VALUES (?,?,?,?)",
                (str(path), title, digest, len(chunks)),
            ).lastrowid
            for i, c in enumerate(chunks):
                blob = vectors[i].tobytes() if vectors is not None else None
                self.conn.execute(
                    "INSERT INTO chunks (doc_id, heading, text, page, position, embedding) VALUES (?,?,?,?,?,?)",
                    (doc_id, c.heading, c.text, c.page, c.position, blob),
                )
        return "updated" if row else "added"

    def ingest_folder(self, folder: Path, embedder=None) -> dict[str, str]:
        results = {}
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                results[p.name] = self.ingest(p, embedder)
        return results

    def remove(self, path: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))

    def documents(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT id, path, title, chunks FROM documents ORDER BY title")]

    def all_chunks(self) -> list[StoredChunk]:
        rows = self.conn.execute(
            "SELECT c.*, d.title AS doc_title, d.path AS doc_path FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id ORDER BY d.title, c.position"
        )
        return [
            StoredChunk(r["id"], r["doc_title"], r["doc_path"], r["heading"], r["text"], r["page"],
                        np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None)
            for r in rows
        ]


__all__ = ["Store", "StoredChunk", "Chunk"]
