import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .answer import Answerer
from .embeddings import from_env
from .loaders import SUPPORTED
from .retriever import Retriever
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(default=4, ge=1, le=8)


def create_app(db_path: str | None = None, docs_dir: str | None = None, ingest_on_start: bool = True) -> FastAPI:
    app = FastAPI(title="Docs RAG Assistant")
    docs_dir = Path(docs_dir or os.getenv("DOCS_DIR", ROOT / "sample_docs"))
    upload_dir = Path(os.getenv("UPLOAD_DIR", ROOT / "uploads"))
    store = Store(db_path or os.getenv("DB_PATH", ROOT / "index.db"))
    embedder = from_env()
    if ingest_on_start and docs_dir.exists():
        store.ingest_folder(docs_dir, embedder)
    retriever = Retriever(store, embedder)
    answerer = Answerer(retriever)
    app.state.store, app.state.retriever, app.state.answerer = store, retriever, answerer

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.post("/api/ask")
    def ask(q: Question):
        a = answerer.ask(q.question, k=q.k)
        return {"answer": a.text, "found": a.found, "mode": a.mode, "sources": a.sources}

    @app.get("/api/search")
    def search(q: str, k: int = 5):
        return [{"citation": h.chunk.citation(), "text": h.chunk.text, "score": round(h.score, 4),
                 "keyword": round(h.keyword, 3), "semantic": h.semantic} for h in retriever.search(q, k)]

    @app.get("/api/documents")
    def documents():
        return {"documents": store.documents(), "embeddings": embedder.name if embedder else None,
                "llm": answerer.provider}

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Supported formats: {', '.join(sorted(SUPPORTED))}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            status = store.ingest(dest, embedder)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f"Couldn't read {name}: {exc}")
        retriever.refresh()
        return {"file": name, "status": status}

    @app.delete("/api/documents/{doc_id}")
    def delete(doc_id: int):
        row = store.conn.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        store.remove(row["path"])
        retriever.refresh()
        return {"deleted": doc_id}

    return app


app = create_app() if os.getenv("RAG_NO_AUTOAPP") != "1" else None
