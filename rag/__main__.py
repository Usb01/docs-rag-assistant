"""Command line:

    python -m rag ingest sample_docs
    python -m rag ask "How many days of annual leave do I get?"
    python -m rag search "service credits"
"""
import argparse
import os
from pathlib import Path

from .answer import Answerer
from .embeddings import from_env
from .retriever import Retriever
from .store import Store


def main():
    ap = argparse.ArgumentParser(prog="rag")
    ap.add_argument("--db", default=os.getenv("DB_PATH", "index.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder", type=Path)
    sub.add_parser("ask").add_argument("question")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    store = Store(args.db)
    embedder = from_env()

    if args.cmd == "ingest":
        for name, status in store.ingest_folder(args.folder, embedder).items():
            print(f"{status:<10} {name}")
        return

    retriever = Retriever(store, embedder)
    if args.cmd == "search":
        for h in retriever.search(args.query, args.k):
            print(f"{h.score:.4f}  {h.chunk.citation()}\n        {h.chunk.text[:140]}")
        return

    a = Answerer(retriever).ask(args.question)
    print(a.text)
    for s in a.sources:
        print(f"  [{s['n']}] {s['citation']}")


if __name__ == "__main__":
    main()
