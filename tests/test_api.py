import os

os.environ["RAG_NO_AUTOAPP"] = "1"

from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from rag.app import create_app  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def client(tmp_path):
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
    return TestClient(create_app(db_path=str(tmp_path / "api.db"), docs_dir=str(DOCS)))


def test_ask_and_documents(tmp_path):
    c = client(tmp_path)
    docs = c.get("/api/documents").json()["documents"]
    assert len(docs) == 4
    r = c.post("/api/ask", json={"question": "How long do secure share links last?"}).json()
    assert r["found"] and "7 days" in r["answer"]


def test_upload_new_document_is_searchable(tmp_path):
    c = client(tmp_path)
    files = {"file": ("parking.md", b"# Parking\n\n## Staff parking\n\nStaff park in zone C behind warehouse 2.", "text/markdown")}
    assert c.post("/api/documents", files=files).json()["status"] == "added"
    r = c.post("/api/ask", json={"question": "Where do staff park?"}).json()
    assert "zone C" in r["answer"]


def test_upload_rejects_other_types(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_delete_document(tmp_path):
    c = client(tmp_path)
    doc_id = c.get("/api/documents").json()["documents"][0]["id"]
    assert c.delete(f"/api/documents/{doc_id}").status_code == 200
    assert len(c.get("/api/documents").json()["documents"]) == 3


def test_eval_set_stays_green():
    import sys
    sys.path.insert(0, str(DOCS.parent / "eval"))
    from run_eval import run
    result = run()
    assert result["hit@3"] >= 0.9 and result["answer_accuracy"] >= 0.85
