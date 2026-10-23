"""Retrieval + answer checks against a fixed question set.

    python eval/run_eval.py            # uses whatever embeddings / LLM the env configures
    EMBEDDINGS=off python eval/run_eval.py

hit@1 / hit@3: was the right section ranked first / in the top 3?
answer: does the answer contain the expected fact (e.g. "21", "60 days")?
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.answer import Answerer  # noqa: E402
from rag.embeddings import from_env  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import Store  # noqa: E402


def run(verbose: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.db")
        embedder = from_env()
        store.ingest_folder(ROOT / "sample_docs", embedder)
        retriever = Retriever(store, embedder)
        answerer = Answerer(retriever)

        rows = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
        hit1 = hit3 = correct = 0
        for row in rows:
            hits = retriever.search(row["q"], k=3)
            ok = [row["doc"] in h.chunk.doc_title and h.chunk.heading == row["heading"] for h in hits]
            hit1 += bool(ok[:1] and ok[0])
            hit3 += any(ok)
            ans = answerer.ask(row["q"])
            good = all(e.lower() in ans.text.lower() for e in row["expect"])
            correct += good
            if verbose and not (any(ok) and good):
                top = hits[0].chunk.citation() if hits else "-"
                print(f"MISS  {row['q']}\n      top: {top}\n      ans: {ans.text[:150]}")

        n = len(rows)
        return {"questions": n, "hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3),
                "answer_accuracy": round(correct / n, 3), "embeddings": embedder.name if embedder else None,
                "llm": answerer.provider}


if __name__ == "__main__":
    print(json.dumps(run(verbose=True), indent=2))
